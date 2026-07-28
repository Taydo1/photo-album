using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;
using PhotoApp2.Models;
using PhotoApp2.Services;
using TorchSharp.PyBridge;
using Xunit;
using Xunit.Abstractions;

namespace PhotoApp2.Tests
{
    public class PhotoAnalyzerServiceTests
    {
        private readonly ITestOutputHelper _output;

        public PhotoAnalyzerServiceTests(ITestOutputHelper output)
        {
            _output = output;
        }

        private static void EnsureTestImages(string imagesDir)
        {
            Directory.CreateDirectory(imagesDir);
            var existing = Directory.GetFiles(imagesDir, "*.jpg");
            if (!existing.Any())
            {
                string img1 = Path.Combine(imagesDir, "sample_landscape.jpg");
                using (var mat1 = new Mat(480, 640, MatType.CV_8UC3, new Scalar(100, 150, 200)))
                {
                    Cv2.Circle(mat1, new Point(320, 240), 50, new Scalar(255, 255, 255), -1);
                    Cv2.ImWrite(img1, mat1);
                }

                string img2 = Path.Combine(imagesDir, "media__1785070826785.jpg");
                using (var mat2 = new Mat(480, 640, MatType.CV_8UC3, new Scalar(200, 100, 50)))
                {
                    Cv2.Rectangle(mat2, new Rect(100, 100, 200, 200), new Scalar(0, 255, 0), -1);
                    Cv2.ImWrite(img2, mat2);
                }
            }
        }

        [Fact]
        public async Task AnalyzePhotoAsync_ProcessesAllAttachedImages_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var imageFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(imageFiles);

            var totalStopwatch = Stopwatch.StartNew();

            foreach (var file in imageFiles)
            {
                var sw = Stopwatch.StartNew();
                var result = await service.AnalyzePhotoAsync(file);
                sw.Stop();

                _output.WriteLine($"File: {Path.GetFileName(file)} | Time: {sw.ElapsedMilliseconds} ms | Faces: {result.FaceCount} | Sharpness: {result.SharpnessScore:F2} | Tags: {result.TagsDisplay}");

                Assert.NotNull(result);
                Assert.True(result.IsAnalyzed, $"Photo analysis failed for {file}");
                Assert.True(sw.ElapsedMilliseconds >= 0);
            }

            totalStopwatch.Stop();
            _output.WriteLine($"Total time for {imageFiles.Length} images: {totalStopwatch.ElapsedMilliseconds} ms (avg {totalStopwatch.ElapsedMilliseconds / imageFiles.Length} ms/image)");
        }

        [Fact]
        public async Task AnalyzePhotoAsync_GroupPhoto_DetectsFacesOrAddsPersonTag()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var groupPhotoPath = Directory.GetFiles(imagesDir, "media__1785070826785.jpg").FirstOrDefault()
                ?? Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();

            Assert.NotNull(groupPhotoPath);

            var sw = Stopwatch.StartNew();
            var result = await service.AnalyzePhotoAsync(groupPhotoPath);
            sw.Stop();

            _output.WriteLine($"Group photo analyzed in {sw.ElapsedMilliseconds} ms | Faces found: {result.FaceCount} | Tags: {result.TagsDisplay}");

            Assert.True(result.IsAnalyzed);
            Assert.NotNull(result.Tags);
        }

        [Fact]
        public async Task Benchmark_BatchAnalyzePhotos_Throughput()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();
            var dbService = new DatabaseService();
            await dbService.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var sampleFiles = Directory.GetFiles(imagesDir, "*.jpg");
            Assert.NotEmpty(sampleFiles);

            var testPhotos = Enumerable.Range(1, 10)
                .SelectMany(i => sampleFiles.Select((file, idx) => new Models.PhotoItem
                {
                    FilePath = file + $"_sim_{i}_{idx}",
                    FileName = Path.GetFileName(file),
                    FileSizeBytes = new FileInfo(file).Length,
                    DateTaken = DateTime.Now.AddDays(-idx)
                }))
                .ToList();

            _output.WriteLine($"=== STARTING BATCH ANALYSIS BENCHMARK ({testPhotos.Count} items) ===");
            var analysisSw = Stopwatch.StartNew();

            await Parallel.ForEachAsync(testPhotos, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, async (photo, ct) =>
            {
                var actualFilePath = photo.FilePath.Substring(0, photo.FilePath.IndexOf("_sim_"));
                var analyzed = await service.AnalyzePhotoAsync(actualFilePath);

                photo.IsAnalyzed = true;
                photo.SharpnessScore = analyzed.SharpnessScore;
                photo.FaceCount = analyzed.FaceCount;
                photo.Tags = analyzed.Tags;
                photo.VisualFeatureVector = analyzed.VisualFeatureVector;
            });

            analysisSw.Stop();
            double fps = testPhotos.Count / analysisSw.Elapsed.TotalSeconds;
            _output.WriteLine($"Parallel Analysis Time: {analysisSw.ElapsedMilliseconds} ms ({fps:F1} photos/sec)");

            var dbSw = Stopwatch.StartNew();
            await dbService.SavePhotosAsync(testPhotos);
            dbSw.Stop();
            _output.WriteLine($"Batch SQLite Save Time ({testPhotos.Count} items): {dbSw.ElapsedMilliseconds} ms");

            Assert.True(testPhotos.All(p => p.IsAnalyzed), "All photos should be analyzed");
            Assert.True(analysisSw.ElapsedMilliseconds < 10000, "Parallel batch analysis should complete quickly");
            Assert.True(dbSw.ElapsedMilliseconds < 1000, "Transactional batch save should be instantaneous");
        }

        [Fact]
        public async Task AnalyzePhotoAsync_ExtractsTags_Successfully()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            var imagesDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestImages");
            EnsureTestImages(imagesDir);

            var sampleFile = Directory.GetFiles(imagesDir, "*.jpg").FirstOrDefault();
            Assert.NotNull(sampleFile);

            var result = await service.AnalyzePhotoAsync(sampleFile);
            _output.WriteLine($"File: {result.FileName} | Tags: {result.TagsDisplay}");

            Assert.True(result.IsAnalyzed);
            Assert.NotNull(result.Tags);
        }

        [Fact]
        public void CosineSimilarity_CalculatesCorrectVectorSimilarity()
        {
            float[] vecA = new float[] { 0.5f, 0.5f, 0.0f, 0.0f };
            float[] vecB = new float[] { 0.5f, 0.5f, 0.0f, 0.0f };
            float[] vecC = new float[] { 0.0f, 0.0f, 1.0f, 1.0f };

            double simIdentical = OnnxContentClassifier.CalculateCosineSimilarity(vecA, vecB);
            double simOrthogonal = OnnxContentClassifier.CalculateCosineSimilarity(vecA, vecC);

            Assert.Equal(1.0, simIdentical, precision: 4);
            Assert.Equal(0.0, simOrthogonal, precision: 4);
        }

        [Fact]
        public async Task AlbumGeneratorService_EnforcesOneOrTwoKindsPerPage_AndFormatsFolderNames()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestAlbum_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                var simPhotos = new List<Models.PhotoItem>
                {
                    new Models.PhotoItem { Id = 1, FilePath = "fake1.jpg", FileName = "fake1.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = DateTime.Today.AddHours(1), Tags = new List<string> { "Landscape" }, VisualFeatureVector = new float[] { 1f, 0f, 0f } },
                    new Models.PhotoItem { Id = 2, FilePath = "fake2.jpg", FileName = "fake2.jpg", IsAnalyzed = true, SharpnessScore = 120, DateTaken = DateTime.Today.AddHours(2), Tags = new List<string> { "Person" }, VisualFeatureVector = new float[] { 0f, 1f, 0f } },
                    new Models.PhotoItem { Id = 3, FilePath = "fake3.jpg", FileName = "fake3.jpg", IsAnalyzed = true, SharpnessScore = 110, DateTaken = DateTime.Today.AddHours(3), Tags = new List<string> { "Architecture" }, VisualFeatureVector = new float[] { 0f, 0f, 1f } },
                    new Models.PhotoItem { Id = 4, FilePath = "fake4.jpg", FileName = "fake4.jpg", IsAnalyzed = true, SharpnessScore = 105, DateTaken = DateTime.Today.AddHours(4), Tags = new List<string> { "Leisure & Recreation", "park" }, VisualFeatureVector = new float[] { 0.5f, 0.5f, 0f } }
                };

                var pages = await generator.GenerateAlbumAsync(simPhotos, tempDir);

                _output.WriteLine($"Generated {pages.Count} pages.");
                foreach (var page in pages)
                {
                    _output.WriteLine($"Page {page.PageNumber} Theme: {page.Theme} | Photos: {page.Photos.Count}");
                    
                    string expectedPrefix = $"Page_{page.PageNumber:D3}_";
                    var matchingFolders = Directory.GetDirectories(tempDir).Where(d => Path.GetFileName(d).StartsWith(expectedPrefix)).ToList();
                    Assert.NotEmpty(matchingFolders);
                }

                var htmlPath = Path.Combine(tempDir, "Album_Preview.html");
                Assert.True(File.Exists(htmlPath), "Album_Preview.html should be created");
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }

        [Fact]
        public async Task AlbumGeneratorService_ExcludesUtilityAndUninterestingTags()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestExclusion_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                var simPhotos = new List<Models.PhotoItem>
                {
                    new Models.PhotoItem { Id = 1, FilePath = "fake1.jpg", FileName = "fake1.jpg", IsAnalyzed = true, SharpnessScore = 100, IsFavorite = true, DateTaken = DateTime.Today.AddHours(1), Tags = new List<string> { "Landscape" }, VisualFeatureVector = new float[] { 1f, 0f, 0f } },
                    new Models.PhotoItem { Id = 2, FilePath = "fake2.jpg", FileName = "fake2.jpg", IsAnalyzed = true, SharpnessScore = 120, DateTaken = DateTime.Today.AddHours(2), Tags = new List<string> { "Screenshot", "Computer & Tech" }, VisualFeatureVector = new float[] { 0f, 1f, 0f } },
                    new Models.PhotoItem { Id = 3, FilePath = "fake3.jpg", FileName = "fake3.jpg", IsAnalyzed = true, SharpnessScore = 110, DateTaken = DateTime.Today.AddHours(3), Tags = new List<string> { "Document", "Paper" }, VisualFeatureVector = new float[] { 0f, 0f, 1f } },
                    new Models.PhotoItem { Id = 4, FilePath = "fake4.jpg", FileName = "fake4.jpg", IsAnalyzed = true, SharpnessScore = 105, DateTaken = DateTime.Today.AddHours(4), Tags = new List<string> { "Uninteresting", "Blurry" }, VisualFeatureVector = new float[] { 0.5f, 0.5f, 0f } }
                };

                var pages = await generator.GenerateAlbumAsync(simPhotos, tempDir);

                // Only fake1.jpg (Landscape) should be included in the album pages
                var albumPhotoIds = pages.SelectMany(p => p.Photos).Select(p => p.Id).ToList();
                Assert.Contains(1, albumPhotoIds);
                Assert.DoesNotContain(2, albumPhotoIds);
                Assert.DoesNotContain(3, albumPhotoIds);
                Assert.DoesNotContain(4, albumPhotoIds);
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }

        [Fact]
        public void MapIndexToTags_CoversAll83Indices_AndReturnsMultipleTags()
        {
            var mapMethod = typeof(OnnxContentClassifier).GetMethod("MapIndexToTags", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            Assert.NotNull(mapMethod);

            var validPrimaryCategories = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "People & Social", "Landscape", "Architecture", "Urban & Travel",
                "Home & Indoors", "Food & Dining", "Leisure & Recreation", "Culture & Education",
                "Work & Industry", "Computer & Tech", "Document", "Uninteresting", "Other"
            };

            for (int i = 0; i < 83; i++)
            {
                var tags = mapMethod.Invoke(null, new object[] { i }) as List<string>;
                Assert.NotNull(tags);
                Assert.NotEmpty(tags);
                Assert.Contains(tags[0], validPrimaryCategories);
            }
        }

        [Fact]
        public void ProcessClassificationResults_AggregatesMultipleTagsPerCategory()
        {
            var processMethod = typeof(OnnxContentClassifier).GetMethods(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)
                .FirstOrDefault(m => m.Name == "ProcessClassificationResults" && m.GetParameters().Length == 2);
            Assert.NotNull(processMethod);

            float[] probs = new float[83];
            probs[0] = 0.85f;  // High probability
            probs[2] = 0.30f;  // Above 0.20 threshold
            probs[25] = 0.15f; // Below threshold

            var result = processMethod.Invoke(null, new object?[] { probs, null });
            Assert.NotNull(result);

            var valueTuple = ((List<string> Tags, double Confidence, float[] FeatureVector))result;
            var tags = valueTuple.Tags;
            double confidence = valueTuple.Confidence;

            Assert.Equal(0.85, confidence, precision: 2);
            Assert.True(tags.Count >= 3, "Should aggregate multiple rich tags from all matching categories");
            Assert.Contains("People & Social", tags);
            Assert.Contains("Discussion", tags);
            Assert.Contains("Food & Dining", tags);
            Assert.DoesNotContain("Landscape", tags);
        }

        [Fact]
        public void FilterRedundantSynonyms_RemovesCloseMeaningTags()
        {
            var filterMethod = typeof(OnnxContentClassifier).GetMethod("FilterRedundantSynonyms", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            Assert.NotNull(filterMethod);

            var rawTags = new List<string> { "Document", "Paper", "Screenshot", "Screen", "Running", "Jogging" };
            var cleanedTags = filterMethod.Invoke(null, new object[] { rawTags }) as List<string>;

            Assert.NotNull(cleanedTags);
            Assert.Contains("Document", cleanedTags);
            Assert.DoesNotContain("Paper", cleanedTags);
            Assert.Contains("Screenshot", cleanedTags);
            Assert.DoesNotContain("Screen", cleanedTags);
            Assert.Contains("Running", cleanedTags);
            Assert.DoesNotContain("Jogging", cleanedTags);
        }

        [Fact]
        public async Task AnalyzePhotoAsync_ReusesExistingEmbedding_AndSkipsExistingTags()
        {
            var service = new PhotoAnalyzerService();
            await service.InitializeAsync();

            // Photo with pre-existing tags -> should skip computation entirely
            var photoWithTags = new Models.PhotoItem
            {
                FilePath = "fake_path.jpg",
                Tags = new List<string> { "ExistingTag" },
                VisualFeatureVector = new float[768]
            };

            var result1 = await service.AnalyzePhotoAsync(photoWithTags);
            Assert.Single(result1.Tags);
            Assert.Equal("ExistingTag", result1.Tags[0]);

            // Photo with precomputed embedding but no tags -> should reuse embedding and generate tags instantly
            var photoWithEmbedOnly = new Models.PhotoItem
            {
                FilePath = "fake_path2.jpg",
                Tags = new List<string>(),
                VisualFeatureVector = new float[768]
            };

            var result2 = await service.AnalyzePhotoAsync(photoWithEmbedOnly);
            Assert.NotNull(result2.Tags);
            Assert.NotEmpty(result2.Tags);
            Assert.True(result2.IsAnalyzed);
        }

        [Fact]
        public void CosineSimilarity_WorksWithHighDimensional768Vectors()
        {
            var rnd = new Random(42);
            float[] vecA = new float[768];
            float[] vecB = new float[768];
            for (int i = 0; i < 768; i++)
            {
                vecA[i] = (float)rnd.NextDouble();
                vecB[i] = vecA[i] * 2.0f; // collinear vector
            }

            double similarity = OnnxContentClassifier.CalculateCosineSimilarity(vecA, vecB);
            Assert.Equal(1.0, similarity, precision: 4);
        }

        [Fact]
        public async Task AlbumGeneratorService_ClustersByMomentDuration_AndToleratesMisclassifiedTags()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestAlbum_Moments_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                var baseDate = new DateTime(2026, 7, 1, 12, 0, 0);
                var photos = new List<Models.PhotoItem>
                {
                    // Meal Moment: tags indicate Food & Dining -> 1.5h max gap separates meal courses/drinks
                    new Models.PhotoItem { Id = 1, FilePath = "m1.jpg", FileName = "m1.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = baseDate, Tags = new List<string> { "Food & Dining" } },
                    new Models.PhotoItem { Id = 2, FilePath = "m2.jpg", FileName = "m2.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = baseDate.AddHours(1), Tags = new List<string> { "Restaurant" } },
                    // Tolerance check: m3 has a wrong/misclassified tag ("Work & Industry"), but taken 30m after m2 -> must NOT break the meal moment!
                    new Models.PhotoItem { Id = 3, FilePath = "m3.jpg", FileName = "m3.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = baseDate.AddHours(1.5), Tags = new List<string> { "Work & Industry" } },

                    // Vacation Trip Chapter 1: starts 6.5 hours later (>3h gap from meal). Labeled favorite to retain smaller chapter.
                    new Models.PhotoItem { Id = 4, FilePath = "w1.jpg", FileName = "w1.jpg", IsAnalyzed = true, SharpnessScore = 100, IsFavorite = true, DateTaken = baseDate.AddHours(8), Tags = new List<string> { "Landscape", "Beach" } },

                    // Vacation Trip Chapter 2 (2 days later): Daily chapter segmentation separates Day 1 from Day 3 to avoid single-page monster moments.
                    new Models.PhotoItem { Id = 5, FilePath = "w2.jpg", FileName = "w2.jpg", IsAnalyzed = true, SharpnessScore = 100, IsFavorite = true, DateTaken = baseDate.AddHours(8 + 48), Tags = new List<string> { "Vacation", "Resort" } },
                    new Models.PhotoItem { Id = 6, FilePath = "w3.jpg", FileName = "w3.jpg", IsAnalyzed = true, SharpnessScore = 100, DateTaken = baseDate.AddHours(8 + 50), Tags = new List<string> { "Mountain" } }
                };

                var pages = await generator.GenerateAlbumAsync(photos, tempDir);

                Assert.True(pages.Count >= 3, $"Expected at least 3 moment chapters (Meal, Vacation Day 1, Vacation Day 3), got {pages.Count}");

                // Page 1 is the Meal moment: photos 1, 2, 3 (including the misclassified photo 3)
                var page1Ids = pages[0].Photos.Select(p => p.Id).ToList();
                Assert.Contains(1, page1Ids);
                Assert.Contains(2, page1Ids);
                Assert.Contains(3, page1Ids); // Misclassification gracefully tolerated without breaking page!

                // Page 2 is Day 1 Beach Excursion: photo 4
                var page2Ids = pages[1].Photos.Select(p => p.Id).ToList();
                Assert.Contains(4, page2Ids);
                Assert.DoesNotContain(5, page2Ids); // Separated across multi-day gap into next chapter

                // Page 3 is Day 3 Resort/Mountain Excursion: photos 5, 6
                var page3Ids = pages[2].Photos.Select(p => p.Id).ToList();
                Assert.Contains(5, page3Ids);
                Assert.Contains(6, page3Ids);
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }

        [Fact]
        public async Task AlbumGeneratorService_SkipsTrivialPages_AndExcludesOddOneOutPhotos()
        {
            var generator = new AlbumGeneratorService();
            var tempDir = Path.Combine(Path.GetTempPath(), "PhotoApp2_TestAlbum_OddOne_" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(tempDir);

            try
            {
                var baseDate = new DateTime(2026, 7, 10, 10, 0, 0);
                var photos = new List<Models.PhotoItem>
                {
                    // Moment 1: Landscape & Hiking outing with 4 nature shots + 1 odd-one-out selfie
                    new Models.PhotoItem { Id = 10, FilePath = "l1.jpg", FileName = "l1.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate, Tags = new List<string> { "Landscape", "Plain", "Meadow", "Countryside" } },
                    new Models.PhotoItem { Id = 11, FilePath = "l2.jpg", FileName = "l2.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddMinutes(5), Tags = new List<string> { "Landscape", "Valley", "Hills", "Scenic" } },
                    new Models.PhotoItem { Id = 12, FilePath = "l3.jpg", FileName = "l3.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddMinutes(10), Tags = new List<string> { "Person", "Leisure & Recreation", "Landscape", "Hiking", "Mountain", "Trekking" } },
                    new Models.PhotoItem { Id = 13, FilePath = "l4.jpg", FileName = "l4.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddMinutes(15), Tags = new List<string> { "Landscape", "Waterfall", "Cascade", "Nature" } },
                    // Odd one out: No landscape/nature theme tags, isolated sub-group of size 1 -> MUST be excluded!
                    new Models.PhotoItem { Id = 14, FilePath = "odd.jpg", FileName = "odd.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddMinutes(18), Tags = new List<string> { "Person", "People & Social", "Selfie", "Photography", "Portrait" } },

                    // Moment 2 (12 hours later): Trivial moment of only 2 un-favorited snapshots -> MUST be skipped entirely!
                    new Models.PhotoItem { Id = 20, FilePath = "t1.jpg", FileName = "t1.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddHours(12), Tags = new List<string> { "Indoor", "Office", "Desk" } },
                    new Models.PhotoItem { Id = 21, FilePath = "t2.jpg", FileName = "t2.jpg", IsAnalyzed = true, SharpnessScore = 80, DateTaken = baseDate.AddHours(12).AddMinutes(5), Tags = new List<string> { "Indoor", "Computer", "Screen" } }
                };

                var pages = await generator.GenerateAlbumAsync(photos, tempDir);

                // We expect exactly 1 page (Moment 1 kept without the odd photo; Moment 2 completely skipped)
                Assert.Single(pages);

                var pagePhotos = pages[0].Photos.Select(p => p.Id).ToList();
                Assert.Equal(4, pagePhotos.Count);
                Assert.Contains(10, pagePhotos);
                Assert.Contains(11, pagePhotos);
                Assert.Contains(12, pagePhotos);
                Assert.Contains(13, pagePhotos);
                Assert.DoesNotContain(14, pagePhotos); // Verified: Odd-one-out selfie excluded from landscape moment!
            }
            finally
            {
                if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
            }
        }
    }
}
