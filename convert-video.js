import ffmpeg from "fluent-ffmpeg";
import { promises as fsPromises } from "node:fs";
import path from "node:path";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import pLimit from "p-limit";

import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import multer from "multer";

dotenv.config();

const app = express();

const NODE_ENV = process.env.NODE_ENV;

const allowedOrigins =
	NODE_ENV === "dev" ? ["http://localhost:3000"] : ["https://gardeai.com"];

app.use(
	cors({
		origin: (origin, callback) => {
			if (!origin || allowedOrigins.includes(origin)) {
				callback(null, true);
			} else {
				callback(new Error("Not allowed by CORS"));
			}
		},
		credentials: true,
		methods: ["POST", "OPTIONS"],
		allowedHeaders: ["Content-Type", "Authorization"],
	}),
);

const PORT = process.env.PORT || 4000;

const upload = multer({ storage: multer.memoryStorage() });

app.options("*", cors());

app.listen(PORT, () => {
	console.log(`Express server running on port ${PORT}`);
});

ffmpeg.setFfmpegPath(process.env.FFMPEG_PATH);
const limit = pLimit(5);

app.post(
	"/api/convert-video/:fencerId?",
	upload.single("video"),
	async (req, res) => {
		try {
			let { fencerId } = req.params;
			if (!fencerId || fencerId === "") {
				fencerId = "no-id";
			}

			const videoFile = req.file;
			if (!videoFile) {
				return res.status(400).json({ message: "No video file uploaded" });
			}

			const videoId = videoFile.originalname;
			const fileExtension = path.extname(videoFile.originalname) || ".unknown";

			const tempInputPath = path.join(
				"/tmp",
				`input_${videoId}.${fileExtension}`,
			);
			const outputDir = path.join("/tmp", `hls_output_${videoId}`);
			await fsPromises.mkdir(outputDir, { recursive: true });

			// Save the video blob to a file
			const arrayBuffer = videoFile.buffer;
			await fsPromises.writeFile(tempInputPath, Buffer.from(arrayBuffer));

			// const metadata = await getVideoMetadata(tempInputPath);
			await uploadMetadata({}, videoId, fencerId);

			try {
				const webmOutputPath = path.join("/tmp", `webm_output_${videoId}.webm`);

				await transcodeToWebM(tempInputPath, webmOutputPath);
				const transcodedBuffer = await fsPromises.readFile(webmOutputPath);

				await uploadToR2(
					transcodedBuffer,
					videoId,
					"full_video.webm",
					fencerId,
					"video/webm",
				);

				await fsPromises.rm(webmOutputPath);
			} catch (error) {
				try {
					console.error(error);
					await uploadToR2(
						arrayBuffer,
						videoId,
						`full_video.${fileExtension}`,
						fencerId,
						videoFile.type,
					);
				} catch (error) {
					console.error(error);
				}
			}

			// If hls conversion fails, upload full video to bucket as failsafe so video is not lost
			let files;
			try {
				await new Promise((resolve, reject) => {
					ffmpeg(tempInputPath)
						.outputOptions([
							"-c:v libx264",
							"-preset fast",
							"-c:a aac",
							"-b:a 128k",
							"-movflags +faststart",
							"-start_number 0",
							"-hls_time 10",
							"-hls_list_size 0",
							"-hls_segment_type mpegts",
							"-f hls",
						])
						.output(path.join(outputDir, "playlist.m3u8"))
						.on("start", (commandLine) => {
							console.log("FFmpeg HLS command:", commandLine);
						})
						.on("stderr", (stderrLine) => {
							console.log("FFmpeg stderr (HLS):", stderrLine);
						})
						.on("end", () => {
							console.log("HLS generation completed");
							resolve();
						})
						.on("error", (error) => {
							console.error("Error generating HLS:", error);
							reject(error);
						})
						.run();
				});

				// Read and upload the .m3u8 and .ts files to Cloudflare R2 or similar storage
				files = await fsPromises.readdir(outputDir);
				if (
					!files.includes("playlist.m3u8") ||
					!files.some((file) => file.endsWith(".ts"))
				) {
					throw new Error(
						"HLS generation failed: Missing required files in outputDir",
					);
				}
			} catch (error) {
				console.error(error);
				return res.status(500).json({ message: "Error converting video" });
			}

			const thumbPath = path.join(outputDir, "thumbnail.jpeg");
			try {
				const len = (files.length - 1) * 10 + 1;
				const timestamp = Math.max(Math.floor(len / 2), 1);

				await generateThumbnail(tempInputPath, thumbPath, timestamp);
				const buff = await fsPromises.readFile(thumbPath);
				await uploadThumbnail(buff, videoId, fencerId);
				await fsPromises.unlink(thumbPath);
			} catch (error) {
				console.error(error);
				await generateThumbnail(tempInputPath, thumbPath, 0);
				const buff = await fsPromises.readFile(thumbPath);
				await uploadThumbnail(buff, videoId, fencerId);
				await fsPromises.unlink(thumbPath);
			}

			const uploadPromises = files.map(async (file) => {
				const filePath = path.join(outputDir, file);
				const fileBuffer = await fsPromises.readFile(filePath);

				await limit(() => uploadToR2(fileBuffer, videoId, file, fencerId));
			});

			await Promise.all(uploadPromises);

			// Clean up temporary files
			await fsPromises.unlink(tempInputPath);
			await Promise.all(
				files.map(async (file) => {
					const filePath = path.join(outputDir, file);
					try {
						await fsPromises.unlink(filePath);
					} catch (error) {
						console.warn(
							`Warning: Could not delete file ${filePath}:`,
							error.message,
						);
					}
				}),
			);

			await fsPromises.rmdir(outputDir);

			return res.status(201).json({ message: "HLS conversion successful" });
		} catch (error) {
			console.error("Error during HLS conversion:", error);
			return res.status(500).json({ message: "Error converting video" });
		}
	},
);

async function transcodeToWebM(inputPath, outputPath) {
	return new Promise((resolve, reject) => {
		ffmpeg(inputPath)
			.outputOptions(["-c:v libvpx", "-b:v 1M", "-c:a libvorbis"])
			.on("end", () => {
				resolve();
			})
			.on("error", (err) => {
				reject(err);
			})
			.save(outputPath);
	});
}

const uploadToR2 = async (
	video,
	videoId,
	segmentId,
	fencerId,
	contentType = null,
) => {
	const r2Client = new S3Client({
		region: "auto",
		endpoint:
			"https://aab5b28251de4c153b96e6f8d3179cbc.r2.cloudflarestorage.com",
		credentials: {
			accessKeyId: process.env.R2_ACCESS_KEY_ID,
			secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
		},
	});

	try {
		let type;
		if (!contentType) {
			if (segmentId.endsWith(".m3u8")) {
				type = "application/vnd.apple.mpegurl";
			} else {
				type = "video/MP2T";
			}
		}

		await r2Client.send(
			new PutObjectCommand({
				Bucket: process.env.BUCKET_NAME,
				Key: `${fencerId}/${videoId}/${segmentId}`,
				Body: video,
				ContentType: contentType || type,
			}),
		);
	} catch (error) {
		console.error("Error uploading to R2:", error);
	}
};

const uploadThumbnail = async (thumbnail, videoId, fencerId) => {
	const r2Client = new S3Client({
		region: "auto",
		endpoint:
			"https://aab5b28251de4c153b96e6f8d3179cbc.r2.cloudflarestorage.com",
		credentials: {
			accessKeyId: process.env.R2_ACCESS_KEY_ID,
			secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
		},
	});

	try {
		await r2Client.send(
			new PutObjectCommand({
				Bucket: process.env.BUCKET_NAME,
				Key: `${fencerId}/${videoId}/thumbnail.jpeg`,
				Body: thumbnail,
				ContentType: "image/jpeg",
			}),
		);
	} catch (error) {
		console.error("Error uploading to R2:", error);
	}
};

const generateThumbnail = async (
	inputPath,
	outputPath,
	timestamp = "00:00:01",
) => {
	try {
		await new Promise((resolve, reject) => {
			ffmpeg(inputPath)
				.inputOptions(["-skip_frame nokey"])
				.seekInput(timestamp)
				.frames(1)
				.outputOptions(["-q:v 2", "-update 1", "-strict unofficial"])
				.output(outputPath)
				.on("start", (commandLine) => {
					console.log("FFmpeg Thumbnail command:", commandLine);
				})
				.on("stderr", (stderrLine) => {
					console.log("FFmpeg stderr (Thumbnail):", stderrLine);
				})
				.on("end", () => {
					console.log("Thumbnail generated:", outputPath);
					resolve(outputPath);
				})
				.on("error", (error) => {
					console.error("Error generating thumbnail:", error);
					reject(error);
				})
				.run();
		});
	} catch (error) {
		console.error("Error in thumbnail generation:", error);
	}
};

const getVideoMetadata = async (videoPath) => {
	return new Promise((resolve, reject) => {
		ffmpeg(videoPath).ffprobe((err, metadata) => {
			if (err) {
				return reject(err);
			}
			resolve(metadata);
		});
	});
};

const uploadMetadata = async (metadata, videoId, fencerId) => {
	const r2Client = new S3Client({
		region: "auto",
		endpoint:
			"https://aab5b28251de4c153b96e6f8d3179cbc.r2.cloudflarestorage.com",
		credentials: {
			accessKeyId: process.env.R2_ACCESS_KEY_ID,
			secretAccessKey: process.env.R2_SECRET_ACCESS_KEY,
		},
	});

	try {
		metadata.timestamp = new Date().toISOString();
		const metadataJson = JSON.stringify(metadata);

		await r2Client.send(
			new PutObjectCommand({
				Bucket: process.env.BUCKET_NAME,
				Key: `${fencerId}/${videoId}/metadata.json`,
				Body: metadataJson,
				ContentType: "application/json",
			}),
		);
	} catch (error) {
		console.error("Error uploading to R2:", error);
	}
};
