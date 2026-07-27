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
        public Task<List<AlbumPage>> GenerateAlbumAsync(IEnumerable<PhotoItem> allPhotos, string destFolderPath, IProgress<string>? progress = null)
        {
            return Task.Run(() =>
            {
                progress?.Report("Filtering and clustering photos by holiday trips and events...");

                var candidates = allPhotos.Where(p => (p.IsAnalyzed && p.SharpnessScore > 50) || p.IsFavorite)
                                          .OrderBy(p => p.DateTaken)
                                          .ToList();

                if (!candidates.Any())
                    return new List<AlbumPage>();

                var episodes = new List<List<PhotoItem>>();
                var currentEpisode = new List<PhotoItem>();
                PhotoItem? lastPhoto = null;

                foreach (var photo in candidates)
                {
                    bool isNewEpisode = false;
                    if (lastPhoto != null)
                    {
                        if ((photo.DateTaken - lastPhoto.DateTaken).TotalHours > 6 || photo.DateTaken.Date != lastPhoto.DateTaken.Date)
                        {
                            isNewEpisode = true;
                        }
                        else if (photo.VisualFeatureVector != null && lastPhoto.VisualFeatureVector != null)
                        {
                            double sim = OnnxContentClassifier.CalculateCosineSimilarity(photo.VisualFeatureVector, lastPhoto.VisualFeatureVector);
                            if (sim < 0.45)
                            {
                                isNewEpisode = true;
                            }
                        }
                    }

                    if (isNewEpisode && currentEpisode.Any())
                    {
                        episodes.Add(currentEpisode);
                        currentEpisode = new List<PhotoItem>();
                    }

                    currentEpisode.Add(photo);
                    lastPhoto = photo;
                }
                if (currentEpisode.Any()) episodes.Add(currentEpisode);

                progress?.Report($"Identified {episodes.Count} chronological & visual episodes. Selecting best photos and deduplicating bursts...");

                var curatedPhotosByEpisode = new List<List<PhotoItem>>();
                foreach (var episode in episodes)
                {
                    var deduplicated = DeduplicateBursts(episode, 15.0);
                    var scored = deduplicated.OrderByDescending(CalculatePhotoScore).ToList();

                    int quota = Math.Min(scored.Count, Math.Max(10, scored.Count * 8 / 10));
                    var selected = scored.Take(quota).OrderBy(p => p.DateTaken).ToList();

                    if (selected.Any()) curatedPhotosByEpisode.Add(selected);
                }

                progress?.Report("Creating rhythm-based thematic pages with visual synergy layout composition...");

                var pages = new List<AlbumPage>();
                int pageCounter = 1;

                foreach (var episodePhotos in curatedPhotosByEpisode)
                {
                    var episodePages = PaginateEpisode(episodePhotos, ref pageCounter);
                    pages.AddRange(episodePages);
                }

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

        private List<AlbumPage> PaginateEpisode(List<PhotoItem> episodePhotos, ref int pageCounter)
        {
            var pages = new List<AlbumPage>();
            var currentPagePhotos = new List<PhotoItem>();

            foreach (var photo in episodePhotos)
            {
                if (currentPagePhotos.Any())
                {
                    var tentativeKinds = currentPagePhotos.Select(p => GetPrimaryTag(p))
                                                          .Concat(new[] { GetPrimaryTag(photo) })
                                                          .Where(k => !string.IsNullOrEmpty(k))
                                                          .Distinct()
                                                          .ToList();

                    bool exceedsKinds = tentativeKinds.Count > 2;

                    int targetCap = DetermineTargetCapacity(currentPagePhotos);
                    bool reachesCap = currentPagePhotos.Count >= targetCap;

                    bool violatesVisualSynergy = false;
                    if (photo.VisualFeatureVector != null && currentPagePhotos.Count >= 2)
                    {
                        double avgSim = currentPagePhotos
                            .Where(p => p.VisualFeatureVector != null)
                            .Select(p => OnnxContentClassifier.CalculateCosineSimilarity(photo.VisualFeatureVector, p.VisualFeatureVector))
                            .DefaultIfEmpty(1.0)
                            .Average();

                        if (avgSim < 0.35)
                        {
                            violatesVisualSynergy = true;
                        }
                    }

                    if (reachesCap || exceedsKinds || violatesVisualSynergy)
                    {
                        pages.Add(CreateAlbumPage(currentPagePhotos, pageCounter++));
                        currentPagePhotos = new List<PhotoItem>();
                    }
                }

                currentPagePhotos.Add(photo);
            }

            if (currentPagePhotos.Any())
            {
                if (currentPagePhotos.Count == 1 && pages.Any())
                {
                    var prevPage = pages.Last();
                    var combinedKinds = prevPage.Photos.Select(p => GetPrimaryTag(p))
                                                 .Concat(currentPagePhotos.Select(p => GetPrimaryTag(p)))
                                                 .Where(k => !string.IsNullOrEmpty(k))
                                                 .Distinct()
                                                 .Count();

                    if (prevPage.Photos.Count < 6 && combinedKinds <= 2)
                    {
                        prevPage.Photos.AddRange(currentPagePhotos);
                        prevPage.Theme = DeterminePageTheme(prevPage.Photos);
                        currentPagePhotos.Clear();
                    }
                }

                if (currentPagePhotos.Any())
                {
                    pages.Add(CreateAlbumPage(currentPagePhotos, pageCounter++));
                }
            }

            return pages;
        }

        private static string GetPrimaryTag(PhotoItem photo)
        {
            if (photo.Tags == null || !photo.Tags.Any()) return "Other";
            return photo.Tags.First();
        }

        private static int DetermineTargetCapacity(List<PhotoItem> pagePhotos)
        {
            if (!pagePhotos.Any()) return 4;

            bool hasHeroCandidate = pagePhotos.Any(p =>
                p.Tags.Any(t => t.Contains("Landscape", StringComparison.OrdinalIgnoreCase) || t.Contains("Architecture", StringComparison.OrdinalIgnoreCase)) &&
                p.SharpnessScore >= 80 && p.FaceCount <= 1);

            if (hasHeroCandidate && pagePhotos.Count <= 2)
            {
                return 2;
            }

            bool isSocialOrEvent = pagePhotos.Any(p =>
                p.FaceCount >= 2 ||
                p.Tags.Any(t => t.Contains("Food", StringComparison.OrdinalIgnoreCase) ||
                                t.Contains("Home", StringComparison.OrdinalIgnoreCase) ||
                                t.Contains("Leisure", StringComparison.OrdinalIgnoreCase) ||
                                t.Contains("Person", StringComparison.OrdinalIgnoreCase)));

            if (isSocialOrEvent)
            {
                return 5;
            }

            return 4;
        }

        private static AlbumPage CreateAlbumPage(List<PhotoItem> photos, int pageNum)
        {
            var page = new AlbumPage
            {
                PageNumber = pageNum,
                Photos = new List<PhotoItem>(photos),
                Theme = DeterminePageTheme(photos)
            };
            return page;
        }

        private static string DeterminePageTheme(List<PhotoItem> photos)
        {
            var kinds = photos.Select(p => GetPrimaryTag(p))
                              .Where(k => !string.IsNullOrEmpty(k))
                              .Distinct()
                              .OrderBy(k => k)
                              .ToList();

            if (!kinds.Any())
                return "Collection";

            return string.Join("_and_", kinds.Select(k => k.Replace(" ", "_")));
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
