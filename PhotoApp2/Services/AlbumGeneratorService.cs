using PhotoApp2.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PhotoApp2.Services
{
    public class AlbumPage
    {
        public int PageNumber { get; set; }
        public string Theme { get; set; } = string.Empty;
        public List<PhotoItem> Photos { get; set; } = new();
        public DateTime StartDate => Photos.Any() ? Photos.Min(p => p.DateTaken) : DateTime.MinValue;
        public DateTime EndDate => Photos.Any() ? Photos.Max(p => p.DateTaken) : DateTime.MinValue;
    }

    public class AlbumGeneratorService
    {
        private static readonly HashSet<string> ExcludedAlbumTags = new(StringComparer.OrdinalIgnoreCase)
        {
            "Screenshot", "Computer & Tech", "Screen", "Monitor",
            "Document", "Paper", "Letter", "Text", "Document Photo",
            "Uninteresting", "Blurry", "Low Quality", "Empty"
        };

        public static bool HasExcludedAlbumTag(PhotoItem photo)
        {
            if (photo.Tags == null || !photo.Tags.Any()) return false;
            return photo.Tags.Any(tag => ExcludedAlbumTags.Contains(tag));
        }

        public Task<List<AlbumPage>> GenerateAlbumAsync(IEnumerable<PhotoItem> allPhotos, string destFolderPath, IProgress<string>? progress = null)
        {
            return Task.Run(() =>
            {
                progress?.Report("Calibrating timestamps and cleaning photo candidates...");

                foreach (var photo in allPhotos)
                {
                    photo.DateTaken = CalibrateDateTaken(photo);
                }

                var candidates = allPhotos.Where(p => ((p.IsAnalyzed && p.SharpnessScore > 50) || p.IsFavorite) && !HasExcludedAlbumTag(p))
                                          .OrderBy(p => p.DateTaken)
                                          .ToList();

                if (!candidates.Any())
                    return new List<AlbumPage>();

                // Step 1: Deduplicate rapid bursts within a 15-second window
                var burstCleaned = DeduplicateBursts(candidates, 15.0);

                // Step 2: Extended Scene Curation (10-minute window) to select representative highlights and eliminate repetitive outtakes
                var sceneCurated = CurateSimilarScenes(burstCleaned, 600.0);

                progress?.Report("Clustering photos into meaningful real-world moments and applying highlight curation...");

                int pageCounter = 1;
                var pages = ClusterIntoMoments(sceneCurated, ref pageCounter);

                progress?.Report($"Exporting {pages.Count} album pages to {destFolderPath}...");

                Directory.CreateDirectory(destFolderPath);
                foreach (var page in pages)
                {
                    string safeTheme = SanitizeFileName(page.Theme);
                    string pageFolderName = $"Page_{page.PageNumber:D3}_{safeTheme}";
                    string pageFolderPath = Path.Combine(destFolderPath, pageFolderName);
                    Directory.CreateDirectory(pageFolderPath);

                    int photoIndex = 1;
                    foreach (var photo in page.Photos)
                    {
                        string destPath = Path.Combine(pageFolderPath, $"{page.PageNumber:D3}_{photoIndex:D2}_{photo.FileName}");
                        try
                        {
                            if (!File.Exists(destPath) && File.Exists(photo.FilePath))
                            {
                                File.Copy(photo.FilePath, destPath);
                            }
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine($"Failed copying file {photo.FilePath} to {destPath}: {ex.Message}");
                        }
                        photoIndex++;
                    }
                }

                GenerateHtmlPreview(pages, destFolderPath);

                progress?.Report($"Album generation complete! Created {pages.Count} pages in {destFolderPath}.");
                return pages;
            });
        }

        private List<PhotoItem> DeduplicateBursts(List<PhotoItem> sortedPhotos, double windowSeconds)
        {
            if (sortedPhotos.Count <= 1) return sortedPhotos;

            var result = new List<PhotoItem>();
            var currentGroup = new List<PhotoItem> { sortedPhotos[0] };

            for (int i = 1; i < sortedPhotos.Count; i++)
            {
                var prevPhoto = sortedPhotos[i - 1];
                var photo = sortedPhotos[i];

                if ((photo.DateTaken - prevPhoto.DateTaken).TotalSeconds <= windowSeconds)
                {
                    currentGroup.Add(photo);
                }
                else
                {
                    ProcessBurstGroup(currentGroup, result);
                    currentGroup.Clear();
                    currentGroup.Add(photo);
                }
            }

            if (currentGroup.Any())
            {
                ProcessBurstGroup(currentGroup, result);
            }

            return result.OrderBy(p => p.DateTaken).ToList();
        }

        private void ProcessBurstGroup(List<PhotoItem> group, List<PhotoItem> result)
        {
            if (group.Count == 1)
            {
                result.Add(group[0]);
                return;
            }

            var retained = new List<PhotoItem>();
            foreach (var photo in group)
            {
                bool isDuplicate = false;
                for (int r = 0; r < retained.Count; r++)
                {
                    var existing = retained[r];
                    double sim = OnnxContentClassifier.CalculateCosineSimilarity(photo.VisualFeatureVector, existing.VisualFeatureVector);
                    if (sim >= 0.92)
                    {
                        isDuplicate = true;
                        if (CalculatePhotoScore(photo) > CalculatePhotoScore(existing))
                        {
                            retained[r] = photo;
                        }
                        break;
                    }
                }

                if (!isDuplicate)
                {
                    retained.Add(photo);
                }
            }

            result.AddRange(retained);
        }

        private static double CalculatePhotoScore(PhotoItem p)
        {
            double score = p.SharpnessScore + (p.FaceCount * 200);
            if (p.IsFavorite) score += 2000;

            if (p.Tags != null && p.Tags.Any())
            {
                if (p.Tags.Any(t => t.Contains("landscape", StringComparison.OrdinalIgnoreCase))) score += 300;
                if (p.Tags.Any(t => t.Contains("architecture", StringComparison.OrdinalIgnoreCase) || t.Contains("landmark", StringComparison.OrdinalIgnoreCase))) score += 200;
                if (p.Tags.Any(t => t.Contains("sunset", StringComparison.OrdinalIgnoreCase) || t.Contains("beach", StringComparison.OrdinalIgnoreCase))) score += 200;
            }

            return score;
        }

        public static DateTime CalibrateDateTaken(PhotoItem photo)
        {
            DateTime current = photo.DateTaken;
            DateTime bestDate = current;

            // 1. Check file modification time (LastWriteTime vs CreationTime) as a physical file fallback
            try
            {
                if (File.Exists(photo.FilePath))
                {
                    var fi = new FileInfo(photo.FilePath);
                    var earliestFileDate = fi.CreationTime < fi.LastWriteTime ? fi.CreationTime : fi.LastWriteTime;
                    if (earliestFileDate < bestDate && earliestFileDate.Year >= 1980 && earliestFileDate.Year <= DateTime.Now.Year + 1)
                    {
                        bestDate = earliestFileDate;
                    }
                }
            }
            catch { }

            // 2. Extract timestamp from filename (e.g. IMG-20180125-WA0017, IMG20210102151606, 20210523_102828)
            var filenameDate = ParseDateFromFileName(photo.FileName, bestDate);
            if (filenameDate.HasValue)
            {
                bestDate = filenameDate.Value;
            }

            return bestDate;
        }

        private static DateTime? ParseDateFromFileName(string fileName, DateTime fallbackTime)
        {
            if (string.IsNullOrWhiteSpace(fileName)) return null;

            try
            {
                string nameWithoutExt = Path.GetFileNameWithoutExtension(fileName);

                // Pattern for yyyyMMddHHmmss (e.g. IMG20210102151606, 20210102_151606, VID_20210102_151606)
                var fullMatch = System.Text.RegularExpressions.Regex.Match(nameWithoutExt, @"(20\d{2})[-_]?((?:0[1-9]|1[0-2]))[-_]?((?:0[1-9]|[12]\d|3[01]))[-_ ]?(([01]\d|2[0-3]))[-_]?(([0-5]\d))[-_]?(([0-5]\d))");
                if (fullMatch.Success)
                {
                    int year = int.Parse(fullMatch.Groups[1].Value);
                    int month = int.Parse(fullMatch.Groups[2].Value);
                    int day = int.Parse(fullMatch.Groups[3].Value);
                    int hour = int.Parse(fullMatch.Groups[4].Value);
                    int min = int.Parse(fullMatch.Groups[6].Value);
                    int sec = int.Parse(fullMatch.Groups[8].Value);
                    return new DateTime(year, month, day, hour, min, sec);
                }

                // Pattern for yyyyMMdd (e.g. IMG-20180125-WA0017, Screenshot_20210815)
                var dateMatch = System.Text.RegularExpressions.Regex.Match(nameWithoutExt, @"(20\d{2})[-_]?((?:0[1-9]|1[0-2]))[-_]?((?:0[1-9]|[12]\d|3[01]))");
                if (dateMatch.Success)
                {
                    int year = int.Parse(dateMatch.Groups[1].Value);
                    int month = int.Parse(dateMatch.Groups[2].Value);
                    int day = int.Parse(dateMatch.Groups[3].Value);

                    return new DateTime(year, month, day, fallbackTime.Hour, fallbackTime.Minute, fallbackTime.Second);
                }
            }
            catch { }

            return null;
        }

        private List<PhotoItem> CurateSimilarScenes(List<PhotoItem> sortedPhotos, double windowSeconds)
        {
            if (sortedPhotos.Count <= 1) return sortedPhotos;

            var result = new List<PhotoItem>();
            var sceneBuffer = new List<PhotoItem>();
            DateTime sceneStartTime = DateTime.MinValue;

            foreach (var photo in sortedPhotos)
            {
                if (!sceneBuffer.Any())
                {
                    sceneBuffer.Add(photo);
                    sceneStartTime = photo.DateTaken;
                    continue;
                }

                string currentParentDir = Path.GetDirectoryName(photo.FilePath) ?? "";
                string lastParentDir = Path.GetDirectoryName(sceneBuffer.Last().FilePath) ?? "";

                if ((photo.DateTaken - sceneStartTime).TotalSeconds <= windowSeconds &&
                    (photo.DateTaken - sceneBuffer.Last().DateTaken).TotalMinutes <= 5.0 &&
                    string.Equals(currentParentDir, lastParentDir, StringComparison.OrdinalIgnoreCase))
                {
                    sceneBuffer.Add(photo);
                }
                else
                {
                    ProcessSceneBuffer(sceneBuffer, result);
                    sceneBuffer.Clear();
                    sceneBuffer.Add(photo);
                    sceneStartTime = photo.DateTaken;
                }
            }

            if (sceneBuffer.Any())
            {
                ProcessSceneBuffer(sceneBuffer, result);
            }

            return result.OrderBy(p => p.DateTaken).ToList();
        }

        private void ProcessSceneBuffer(List<PhotoItem> buffer, List<PhotoItem> result)
        {
            if (buffer.Count <= 2)
            {
                result.AddRange(buffer);
                return;
            }

            var selected = new List<PhotoItem>();
            foreach (var photo in buffer.OrderByDescending(CalculatePhotoScore))
            {
                if (photo.IsFavorite)
                {
                    selected.Add(photo);
                    continue;
                }

                bool isTooSimilar = false;
                foreach (var existing in selected)
                {
                    double sim = OnnxContentClassifier.CalculateCosineSimilarity(photo.VisualFeatureVector, existing.VisualFeatureVector);
                    string tag1 = GetPrimaryTag(photo);
                    string tag2 = GetPrimaryTag(existing);

                    // Only discard genuine visual near-duplicates or redundant successive attempts
                    if (sim >= 0.85 || (sim >= 0.78 && tag1 == tag2 && tag1 != "Other" && photo.FaceCount == existing.FaceCount))
                    {
                        isTooSimilar = true;
                        break;
                    }
                }

                if (!isTooSimilar)
                {
                    selected.Add(photo);
                }
            }

            result.AddRange(selected.OrderBy(p => p.DateTaken));
        }

        private List<AlbumPage> ClusterIntoMoments(List<PhotoItem> sortedPhotos, ref int pageCounter)
        {
            var pages = new List<AlbumPage>();
            var currentMomentPhotos = new List<PhotoItem>();

            foreach (var photo in sortedPhotos)
            {
                if (currentMomentPhotos.Any())
                {
                    if (ShouldSplitMoment(currentMomentPhotos, photo))
                    {
                        var filtered = FilterOddOneOutPhotos(currentMomentPhotos);
                        if (filtered.Count >= 3 || filtered.Any(p => p.IsFavorite))
                        {
                            pages.Add(CreateAlbumPage(filtered, pageCounter++));
                        }
                        currentMomentPhotos = new List<PhotoItem>();
                    }
                }

                currentMomentPhotos.Add(photo);
            }

            if (currentMomentPhotos.Any())
            {
                var filtered = FilterOddOneOutPhotos(currentMomentPhotos);
                if (filtered.Count >= 3 || filtered.Any(p => p.IsFavorite))
                {
                    pages.Add(CreateAlbumPage(filtered, pageCounter++));
                }
            }

            return pages;
        }

        private static List<PhotoItem> FilterOddOneOutPhotos(List<PhotoItem> photos)
        {
            if (photos.Count <= 2) return photos;

            // Collect thematic tags across the moment, ignoring super-generic tags that don't define a specific activity scene
            var genericTags = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Person", "Other", "People & Social", "Photography", "Selfie", "Portrait", "Photo", "Image", "Collection", "Leisure & Recreation"
            };

            var photoThemes = new Dictionary<PhotoItem, HashSet<string>>();
            var themeFrequencies = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

            foreach (var p in photos)
            {
                var tags = (p.Tags ?? new List<string>())
                            .Where(t => !string.IsNullOrWhiteSpace(t) && !genericTags.Contains(t))
                            .ToHashSet(StringComparer.OrdinalIgnoreCase);

                photoThemes[p] = tags;
                foreach (var tag in tags)
                {
                    themeFrequencies[tag] = themeFrequencies.GetValueOrDefault(tag) + 1;
                }
            }

            // A sub-group theme is dominant/valid if several (>= 2) photos in the moment share that theme
            int minSubGroupSize = photos.Count >= 5 ? 2 : 1;
            var dominantThemes = themeFrequencies.Where(kv => kv.Value >= minSubGroupSize)
                                                 .Select(kv => kv.Key)
                                                 .ToHashSet(StringComparer.OrdinalIgnoreCase);

            if (!dominantThemes.Any()) return photos;

            var filtered = new List<PhotoItem>();
            foreach (var photo in photos)
            {
                if (photo.IsFavorite)
                {
                    filtered.Add(photo);
                    continue;
                }

                var myTags = photoThemes[photo];
                bool matchesDominantTheme = myTags.Any(t => dominantThemes.Contains(t));

                if (matchesDominantTheme)
                {
                    filtered.Add(photo);
                }
                else
                {
                    // Check if this photo has several similar photos forming a coherent sub-group within the moment
                    int similarSubGroupCount = photos.Count(other =>
                        other == photo ||
                        (photoThemes[other].Any() && photoThemes[other].Any(t => myTags.Contains(t))) ||
                        OnnxContentClassifier.CalculateCosineSimilarity(photo.VisualFeatureVector, other.VisualFeatureVector) >= 0.78
                    );

                    if (similarSubGroupCount >= 3 || (photos.Count <= 4 && similarSubGroupCount >= 2))
                    {
                        filtered.Add(photo);
                    }
                    else
                    {
                        Debug.WriteLine($"Excluding odd-one-out photo {photo.FileName} from moment.");
                    }
                }
            }

            return filtered.Any() ? filtered : photos;
        }

        private static bool ShouldSplitMoment(List<PhotoItem> currentPhotos, PhotoItem candidate)
        {
            if (!currentPhotos.Any()) return false;

            var lastPhoto = currentPhotos.Last();
            var firstPhoto = currentPhotos.First();

            double gapHours = (candidate.DateTaken - lastPhoto.DateTaken).TotalHours;
            double totalSpanHours = (candidate.DateTaken - firstPhoto.DateTaken).TotalHours;

            // 0. Hard separation: overnight intermissions (> 5 hours) or calendar day transitions (> 3 hours gap) ALWAYS separate moments.
            if (gapHours > 5.0 || (candidate.DateTaken.Date != lastPhoto.DateTaken.Date && gapHours > 3.0))
            {
                return true;
            }

            // 1. Determine moment duration thresholds based on semantic tags of the current moment
            GetMomentTimeThresholds(currentPhotos, out double maxGapHours, out double maxSpanHours, out _);

            if (gapHours > maxGapHours || totalSpanHours > maxSpanHours)
            {
                return true; // Chronological boundary exceeded for this activity type
            }

            // 2. Time-proximity noise tolerance: Within 25 minutes (0.42 hours), never break on tag divergence or classification noise.
            if (gapHours <= 0.42)
            {
                return false;
            }

            // 3. For moderate intervals within the allowed threshold (> 25 mins but <= maxGapHours),
            // check for a sustained thematic divergence rather than an isolated outlier tag.
            string candidateTag = GetPrimaryTag(candidate);
            var dominantTags = GetDominantTags(currentPhotos);

            if (!dominantTags.Contains(candidateTag, StringComparer.OrdinalIgnoreCase))
            {
                int existingOutliers = currentPhotos.Count(p => !dominantTags.Contains(GetPrimaryTag(p), StringComparer.OrdinalIgnoreCase));
                if (existingOutliers + 1 > Math.Max(1, currentPhotos.Count * 0.25))
                {
                    return true;
                }
            }

            return false;
        }

        private static void GetMomentTimeThresholds(IEnumerable<PhotoItem> momentPhotos, out double maxGapHours, out double maxSpanHours, out int maxHighlights)
        {
            var allTags = momentPhotos.SelectMany(p => p.Tags ?? new List<string>()).Where(t => !string.IsNullOrEmpty(t)).ToList();
            if (!allTags.Any())
            {
                maxGapHours = 3.0;   // 3 hours intermission separates daily excursions
                maxSpanHours = 8.0;  // 8 hours max per daytime event chapter
                maxHighlights = 20;  // 20 curated highlights maximum
                return;
            }

            int mealTags = allTags.Count(t => ContainsAny(t, "Food", "Dining", "Drink", "Restaurant", "Cafe", "Meal", "Kitchen", "Bar", "Beverage"));
            int total = allTags.Count;

            // Meal or dining gathering
            if (mealTags > 0 && (mealTags * 100.0 / total) >= 30)
            {
                maxGapHours = 1.5;  // 1.5 hours gap (e.g., between appetizers and dessert/after-drinks)
                maxSpanHours = 3.5; // 3.5 hours max duration for a banquet or dinner gathering
                maxHighlights = 10; // 10 highlights maximum
                return;
            }

            // Outing, activity, landscape, resort, or scenic excursion chapter
            maxGapHours = 3.0;   // 3 hours intermission separates major activities on a trip
            maxSpanHours = 8.0;  // 8 hours max contiguous span per activity chapter
            maxHighlights = 25;  // 25 curated highlights maximum per activity moment
        }

        private static bool ContainsAny(string tag, params string[] keywords)
        {
            foreach (var kw in keywords)
            {
                if (tag.IndexOf(kw, StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            }
            return false;
        }

        private static string GetPrimaryTag(PhotoItem photo)
        {
            if (photo.Tags == null || !photo.Tags.Any()) return "Other";
            return photo.Tags.First();
        }

        private static AlbumPage CreateAlbumPage(List<PhotoItem> photos, int pageNum)
        {
            var curated = CurateMomentHighlights(photos);
            var page = new AlbumPage
            {
                PageNumber = pageNum,
                Photos = new List<PhotoItem>(curated),
                Theme = DeterminePageTheme(curated)
            };
            return page;
        }

        private static List<PhotoItem> CurateMomentHighlights(List<PhotoItem> momentPhotos)
        {
            if (momentPhotos.Count <= 12) return momentPhotos;

            GetMomentTimeThresholds(momentPhotos, out _, out _, out int maxHighlights);
            if (momentPhotos.Count <= maxHighlights) return momentPhotos;

            var favorites = momentPhotos.Where(p => p.IsFavorite).ToList();
            int remainingQuota = Math.Max(4, maxHighlights - favorites.Count);

            var bestRemaining = momentPhotos.Where(p => !p.IsFavorite)
                                            .OrderByDescending(CalculatePhotoScore)
                                            .Take(remainingQuota)
                                            .ToList();

            return favorites.Concat(bestRemaining)
                            .Distinct()
                            .OrderBy(p => p.DateTaken)
                            .ToList();
        }

        private static List<string> GetDominantTags(IEnumerable<PhotoItem> photos)
        {
            var tagCounts = photos.Select(p => GetPrimaryTag(p))
                                  .Where(k => !string.IsNullOrEmpty(k) && k != "Other" && k != "Collection")
                                  .GroupBy(k => k, StringComparer.OrdinalIgnoreCase)
                                  .Select(g => new { Tag = g.Key, Count = g.Count() })
                                  .OrderByDescending(x => x.Count)
                                  .ToList();

            if (!tagCounts.Any())
                return new List<string> { "Collection" };

            // Select top 1 or 2 dominant categories, effectively ignoring noise or misclassified tags
            return tagCounts.Take(2).Select(x => x.Tag).OrderBy(x => x).ToList();
        }

        private static string DeterminePageTheme(List<PhotoItem> photos)
        {
            var dominant = GetDominantTags(photos);
            if (!dominant.Any() || (dominant.Count == 1 && dominant[0] == "Collection"))
                return "Collection";

            return string.Join("_and_", dominant.Select(k => k.Replace(" ", "_")));
        }

        private static string SanitizeFileName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var sb = new StringBuilder();
            foreach (var ch in name)
            {
                if (!invalid.Contains(ch) && ch != '/' && ch != '\\')
                    sb.Append(ch);
                else
                    sb.Append('_');
            }
            return sb.ToString().Trim('_');
        }

        private static void GenerateHtmlPreview(List<AlbumPage> pages, string destFolderPath)
        {
            var html = new StringBuilder();
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html lang='en'><head><meta charset='utf-8'><title>Intelligent Photo Album Preview</title>");
            html.AppendLine("<style>");
            html.AppendLine(":root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #38bdf8; --border: #334155; }");
            html.AppendLine("body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: var(--bg); color: var(--text); margin: 0; padding: 40px 20px; }");
            html.AppendLine(".container { max-width: 1200px; margin: 0 auto; }");
            html.AppendLine("h1 { text-align: center; font-weight: 700; color: var(--accent); margin-bottom: 10px; font-size: 2.5em; letter-spacing: -0.5px; }");
            html.AppendLine(".summary { text-align: center; color: #94a3b8; margin-bottom: 40px; font-size: 1.1em; }");
            html.AppendLine(".page-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 25px; margin-bottom: 35px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); transition: transform 0.2s; }");
            html.AppendLine(".page-card:hover { transform: translateY(-3px); }");
            html.AppendLine(".page-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 15px; }");
            html.AppendLine(".page-title { font-size: 1.5em; font-weight: 600; color: var(--text); display: flex; align-items: center; gap: 12px; }");
            html.AppendLine(".badge { background: var(--accent); color: #000; padding: 4px 12px; border-radius: 20px; font-size: 0.75em; font-weight: 700; text-transform: uppercase; }");
            html.AppendLine(".page-meta { color: #94a3b8; font-size: 0.9em; }");
            html.AppendLine(".photo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 15px; }");
            html.AppendLine(".photo-item { background: #000; border-radius: 10px; overflow: hidden; position: relative; aspect-ratio: 4/3; }");
            html.AppendLine(".photo-item img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.3s ease; }");
            html.AppendLine(".photo-item:hover img { transform: scale(1.05); }");
            html.AppendLine(".photo-tag { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0, 0, 0, 0.75); backdrop-filter: blur(4px); color: #e2e8f0; font-size: 0.8em; padding: 8px 12px; box-sizing: border-box; }");
            html.AppendLine("</style></head><body>");
            html.AppendLine("<div class='container'>");
            html.AppendLine("<h1>✨ AI Curated Photo Album</h1>");
            html.AppendLine($"<div class='summary'>Generated on {DateTime.Now:MMMM dd, yyyy} &bull; {pages.Count} Chronological & Thematic Pages &bull; {pages.Sum(p => p.Photos.Count)} Total Photos</div>");

            foreach (var page in pages)
            {
                string safeTheme = SanitizeFileName(page.Theme);
                string pageFolderName = $"Page_{page.PageNumber:D3}_{safeTheme}";
                string readableTheme = page.Theme.Replace("_and_", " & ").Replace("_", " ");

                html.AppendLine("<div class='page-card'>");
                html.AppendLine("<div class='page-header'>");
                html.AppendLine($"<div class='page-title'>Page {page.PageNumber} <span class='badge'>{readableTheme}</span></div>");
                string dateStr = page.StartDate.Date == page.EndDate.Date 
                    ? page.StartDate.ToString("MMM dd, yyyy") 
                    : $"{page.StartDate:MMM dd} &ndash; {page.EndDate:MMM dd, yyyy}";
                html.AppendLine($"<div class='page-meta'>{dateStr} &bull; {page.Photos.Count} Photos</div>");
                html.AppendLine("</div>");
                html.AppendLine("<div class='photo-grid'>");

                int pIdx = 1;
                foreach (var photo in page.Photos)
                {
                    string relPath = $"{pageFolderName}/{page.PageNumber:D3}_{pIdx:D2}_{photo.FileName}";
                    string caption = photo.TagsDisplay;
                    html.AppendLine($"<div class='photo-item' title='{photo.FileName} &bull; {caption}'>");
                    html.AppendLine($"<img src='{relPath}' alt='{photo.FileName}' loading='lazy'/>");
                    html.AppendLine($"<div class='photo-tag'>{caption}</div>");
                    html.AppendLine("</div>");
                    pIdx++;
                }

                html.AppendLine("</div></div>");
            }

            html.AppendLine("</div></body></html>");
            try
            {
                File.WriteAllText(Path.Combine(destFolderPath, "Album_Preview.html"), html.ToString(), Encoding.UTF8);
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to write HTML preview: {ex.Message}");
            }
        }
    }
}
