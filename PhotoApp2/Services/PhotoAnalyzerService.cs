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
                    DateTaken = fileInfo.CreationTime, // Default fallback
                };

                try
                {
                    // 1. Fast EXIF extraction via Ping (header only, no pixel decode/re-encode)
                    var exifDate = ExtractExifDateTaken(filePath);
                    if (exifDate.HasValue)
                    {
                        photo.DateTaken = exifDate.Value;
                    }

                    // 2. Shared visual analysis without thread hopping (OpenCV BGR Mat used for Sharpness, GPU Face Detection, and GPU Scene Classification)
                    var (sharpness, faces, keywords, primaryKind) = AnalyzeVisuals(filePath);
                    photo.SharpnessScore = sharpness;
                    photo.FaceCount = faces;
                    photo.Keywords = keywords;
                    photo.PrimaryKind = primaryKind;

                    // 3. Scene Classification
                    if (photo.FaceCount > 0 && photo.PrimaryKind != "Landscape")
                    {
                        photo.SceneCategory = "Person";
                    }
                    else if (!string.IsNullOrEmpty(photo.PrimaryKind))
                    {
                        photo.SceneCategory = photo.PrimaryKind;
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
            });
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

        private (double Sharpness, int Faces, string Keywords, string PrimaryKind) AnalyzeVisuals(string filePath)
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

                    string sceneKeywords = "";
                    string sceneKind = "Other";
                    if (_contentClassifier != null)
                    {
                        var sceneRes = _contentClassifier.ClassifyScene(mat);
                        sceneKeywords = sceneRes.Keywords;
                        sceneKind = sceneRes.PrimaryKind;
                    }

                    return SynthesizeKeywords(faces, sceneKeywords, sceneKind, sharpness);
                }
            }
            catch { }

            // Fallback for non-standard image formats (e.g. HEIC/RAW) via ImageMagick transcode
            double fallbackSharpness = 0;
            int fallbackFaces = 0;
            string fallbackKeywords = "";
            string fallbackKind = "Other";

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
                        fallbackKeywords = sceneRes.Keywords;
                        fallbackKind = sceneRes.PrimaryKind;
                    }
                }
            }
            catch { }

            return SynthesizeKeywords(fallbackFaces, fallbackKeywords, fallbackKind, fallbackSharpness);
        }

        private static (double Sharpness, int Faces, string Keywords, string PrimaryKind) SynthesizeKeywords(int faces, string sceneKeywords, string sceneKind, double sharpness)
        {
            string keywords = sceneKeywords ?? "";
            string primaryKind = sceneKind ?? "Other";

            if (faces > 0)
            {
                var faceTag = faces >= 3 ? "group portrait, person" : "person";
                keywords = string.IsNullOrEmpty(keywords) ? faceTag : $"{faceTag}, {keywords}";
                
                if (primaryKind == "Landscape" || primaryKind == "Architecture")
                {
                    primaryKind = $"{primaryKind}_and_Person";
                }
                else if (primaryKind == "Other" || primaryKind == "Animal" || string.IsNullOrEmpty(primaryKind))
                {
                    primaryKind = "Person";
                }
            }
            else if (string.IsNullOrEmpty(primaryKind))
            {
                primaryKind = "Other";
            }

            return (sharpness, faces, keywords, primaryKind);
        }
    }
}
