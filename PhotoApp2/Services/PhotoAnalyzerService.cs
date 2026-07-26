using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using OpenCvSharp;
using PhotoApp2.Models;
using ImageMagick;

namespace PhotoApp2.Services
{
    public class PhotoAnalyzerService
    {
        private OnnxFaceDetector? _faceDetector;

        public async Task InitializeAsync()
        {
            if (_faceDetector == null)
            {
                try
                {
                    _faceDetector = new OnnxFaceDetector();
                    await _faceDetector.InitializeAsync();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxFaceDetector: {ex.Message}");
                }
            }
        }

        public async Task<PhotoItem> AnalyzePhotoAsync(string filePath)
        {
            var fileInfo = new FileInfo(filePath);
            var photo = new PhotoItem
            {
                FilePath = filePath,
                FileName = fileInfo.Name,
                FileSizeBytes = fileInfo.Length,
                DateTaken = fileInfo.CreationTime, // Default fallback
            };

            try
            {
                // Task 1: Fast EXIF extraction via Ping (header only, no pixel decode/re-encode)
                var exifTask = Task.Run(() => ExtractExifDateTaken(filePath));

                // Task 2: Shared visual analysis (OpenCV Grayscale Mat used for both Sharpness and Face Detection)
                var visualTask = AnalyzeVisualsAsync(filePath);

                // Run EXIF extraction and visual analysis concurrently
                await Task.WhenAll(exifTask, visualTask);

                if (exifTask.Result.HasValue)
                {
                    photo.DateTaken = exifTask.Result.Value;
                }

                var visualResult = visualTask.Result;
                photo.SharpnessScore = visualResult.Sharpness;
                photo.FaceCount = visualResult.Faces;

                // 3. Scene Classification
                if (photo.FaceCount > 0)
                {
                    photo.SceneCategory = "Person";
                }
                else
                {
                    photo.SceneCategory = "Landscape / Other";
                }

                photo.IsAnalyzed = true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Error analyzing photo {filePath}: {ex.Message}");
                photo.IsAnalyzed = false;
            }

            return photo;
        }

        private static DateTime? ExtractExifDateTaken(string filePath)
        {
            try
            {
                // Ping header only without decompressing pixel data
                using var magickImage = new MagickImage();
                magickImage.Ping(filePath);
                
                var exifProfile = magickImage.GetExifProfile();
                if (exifProfile != null)
                {
                    var dateTakenTag = exifProfile.GetValue(ExifTag.DateTimeOriginal);
                    if (dateTakenTag != null && DateTime.TryParseExact(
                        dateTakenTag.ToString(), 
                        "yyyy:MM:dd HH:mm:ss", 
                        null, 
                        System.Globalization.DateTimeStyles.None, 
                        out var dateTaken))
                    {
                        return dateTaken;
                    }
                }
            }
            catch
            {
                // Fallback date taken remains FileInfo.CreationTime
            }

            return null;
        }

        private async Task<(double Sharpness, int Faces)> AnalyzeVisualsAsync(string filePath)
        {
            double sharpness = 0;
            int faces = 0;

            try
            {
                // Decode once into native Grayscale and run high-speed sharpness + GPU ONNX face detection
                using var mat = await Task.Run(() => Cv2.ImRead(filePath, ImreadModes.Grayscale));
                if (!mat.Empty())
                {
                    await Task.Run(() =>
                    {
                        using var laplacian = new Mat();
                        Cv2.Laplacian(mat, laplacian, MatType.CV_64F);
                        Cv2.MeanStdDev(laplacian, out _, out var stddev);
                        sharpness = stddev.Val0 * stddev.Val0;
                    });

                    if (_faceDetector != null)
                    {
                        faces = await _faceDetector.DetectFacesAsync(mat);
                    }
                    return (sharpness, faces);
                }
            }
            catch { }

            // Fallback for non-standard image formats (e.g. HEIC/RAW) via ImageMagick transcode
            try
            {
                using var magickImage = new MagickImage(filePath);
                using var memStream = new MemoryStream();
                magickImage.Format = MagickFormat.Jpeg;
                magickImage.Write(memStream);
                
                using var fallbackMat = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Grayscale);
                if (!fallbackMat.Empty() && _faceDetector != null)
                {
                    faces = await _faceDetector.DetectFacesAsync(fallbackMat);
                }
            }
            catch { }

            return (sharpness, faces);
        }
    }
}
