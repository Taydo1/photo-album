using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
        private OnnxContentClassifier? _contentClassifier;

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
            if (_contentClassifier == null)
            {
                try
                {
                    _contentClassifier = new OnnxContentClassifier();
                    await _contentClassifier.InitializeAsync();
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error initializing OnnxContentClassifier: {ex.Message}");
                }
            }
        }

        public Task<PhotoItem> AnalyzePhotoAsync(PhotoItem photo)
        {
            return Task.Run(() =>
            {
                if (photo == null) return photo!;

                // 1. If tags already exist on the photo, skip tag computation completely
                if (photo.Tags != null && photo.Tags.Any())
                {
                    photo.IsAnalyzed = true;
                    return photo;
                }

                // 2. If feature vector already exists, reuse it without re-running ONNX vision model
                if (photo.VisualFeatureVector != null && photo.VisualFeatureVector.Length > 0 && _contentClassifier != null)
                {
                    var res = _contentClassifier.ClassifyFromEmbedding(photo.VisualFeatureVector);
                    var finalTags = new List<string>(res.Tags ?? new List<string>());
                    if (photo.FaceCount > 0 && !finalTags.Contains("Person", StringComparer.OrdinalIgnoreCase))
                    {
                        finalTags.Insert(0, "Person");
                    }
                    photo.Tags = finalTags;
                    photo.IsAnalyzed = true;
                    return photo;
                }

                // 3. Fallback to full visual analysis from file path
                var fullAnalyzed = AnalyzePhotoAsync(photo.FilePath).Result;
                photo.SharpnessScore = fullAnalyzed.SharpnessScore;
                photo.FaceCount = fullAnalyzed.FaceCount;
                photo.Tags = fullAnalyzed.Tags;
                photo.VisualFeatureVector = fullAnalyzed.VisualFeatureVector;
                photo.IsAnalyzed = true;
                return photo;
            });
        }

        public Task<PhotoItem> AnalyzePhotoAsync(string filePath)
        {
            return Task.Run(() =>
            {
                var fileInfo = new FileInfo(filePath);
                var photo = new PhotoItem
                {
                    FilePath = filePath,
                    FileName = fileInfo.Name,
                    FileSizeBytes = fileInfo.Length,
                    DateTaken = fileInfo.CreationTime < fileInfo.LastWriteTime ? fileInfo.CreationTime : fileInfo.LastWriteTime, // Robust default fallback
                };

                try
                {
                    var exifDate = ExtractExifDateTaken(filePath);
                    if (exifDate.HasValue)
                    {
                        photo.DateTaken = exifDate.Value;
                    }
                    else
                    {
                        photo.DateTaken = AlbumGeneratorService.CalibrateDateTaken(photo);
                    }

                    var (sharpness, faces, tags, featureVector) = AnalyzeVisuals(filePath);
                    photo.SharpnessScore = sharpness;
                    photo.FaceCount = faces;
                    photo.VisualFeatureVector = featureVector;

                    // Add "Person" to tags if a face is detected
                    var finalTags = new List<string>(tags ?? new List<string>());
                    if (photo.FaceCount > 0 && !finalTags.Contains("Person", StringComparer.OrdinalIgnoreCase))
                    {
                        finalTags.Insert(0, "Person");
                    }
                    photo.Tags = finalTags;

                    photo.IsAnalyzed = true;
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Error analyzing photo {filePath}: {ex.Message}");
                    photo.IsAnalyzed = false;
                }

                return photo;
            });
        }

        private static DateTime? ExtractExifDateTaken(string filePath)
        {
            try
            {
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
            catch { }

            return null;
        }

        private (double Sharpness, int Faces, List<string> Tags, float[]? FeatureVector) AnalyzeVisuals(string filePath)
        {
            try
            {
                using var mat = Cv2.ImRead(filePath, ImreadModes.Color);
                if (!mat.Empty())
                {
                    double sharpness = 0;
                    using (var gray = new Mat())
                    using (var laplacian = new Mat())
                    {
                        Cv2.CvtColor(mat, gray, ColorConversionCodes.BGR2GRAY);
                        Cv2.Laplacian(gray, laplacian, MatType.CV_32F);
                        Cv2.MeanStdDev(laplacian, out _, out var stddev);
                        sharpness = stddev.Val0 * stddev.Val0;
                    }

                    int faces = 0;
                    if (_faceDetector != null)
                    {
                        faces = _faceDetector.DetectFaces(mat);
                    }

                    List<string> sceneTags = new();
                    float[]? featureVec = null;

                    if (_contentClassifier != null)
                    {
                        var sceneRes = _contentClassifier.ClassifyScene(mat);
                        sceneTags = sceneRes.Tags;
                        featureVec = sceneRes.FeatureVector;
                    }

                    return (sharpness, faces, sceneTags, featureVec);
                }
            }
            catch { }

            double fallbackSharpness = 0;
            int fallbackFaces = 0;
            List<string> fallbackTags = new();
            float[]? fallbackFeatureVec = null;

            try
            {
                using var magickImage = new MagickImage(filePath);
                using var memStream = new MemoryStream();
                magickImage.Format = MagickFormat.Jpeg;
                magickImage.Write(memStream);
                
                using var fallbackMat = Cv2.ImDecode(memStream.ToArray(), ImreadModes.Color);
                if (!fallbackMat.Empty())
                {
                    using (var gray = new Mat())
                    using (var laplacian = new Mat())
                    {
                        Cv2.CvtColor(fallbackMat, gray, ColorConversionCodes.BGR2GRAY);
                        Cv2.Laplacian(gray, laplacian, MatType.CV_32F);
                        Cv2.MeanStdDev(laplacian, out _, out var stddev);
                        fallbackSharpness = stddev.Val0 * stddev.Val0;
                    }
                    if (_faceDetector != null)
                    {
                        fallbackFaces = _faceDetector.DetectFaces(fallbackMat);
                    }
                    if (_contentClassifier != null)
                    {
                        var sceneRes = _contentClassifier.ClassifyScene(fallbackMat);
                        fallbackTags = sceneRes.Tags;
                        fallbackFeatureVec = sceneRes.FeatureVector;
                    }
                }
            }
            catch { }

            return (fallbackSharpness, fallbackFaces, fallbackTags, fallbackFeatureVec);
        }
    }
}
