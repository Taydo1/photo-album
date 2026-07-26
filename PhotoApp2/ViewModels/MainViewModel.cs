using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PhotoApp2.Models;
using PhotoApp2.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Windows.Storage.Pickers;
using WinRT.Interop;

namespace PhotoApp2.ViewModels
{
    public partial class MainViewModel : ObservableObject
    {
        private readonly DatabaseService _dbService;
        private readonly PhotoAnalyzerService _analyzerService;

        [ObservableProperty]
        private ObservableCollection<PhotoItem> _photos = new();

        [ObservableProperty]
        private ObservableCollection<FolderNode> _folderNodes = new();

        [ObservableProperty]
        private FolderNode? _selectedFolderNode;

        [ObservableProperty]
        private bool _isAnalyzing;

        [ObservableProperty]
        private string _statusMessage = "Ready";

        [ObservableProperty]
        private int _totalPhotos;

        [ObservableProperty]
        private int _analyzedPhotos;
        
        [ObservableProperty]
        private PhotoItem? _selectedPhoto;

        // Filtering
        [ObservableProperty]
        private bool _showFavoritesOnly;

        [ObservableProperty]
        private bool _showPeopleOnly;

        [ObservableProperty]
        private bool _isGalleryVisible = true;
        
        private List<PhotoItem> _allPhotosCache = new();

        public MainViewModel(DatabaseService dbService, PhotoAnalyzerService analyzerService)
        {
            _dbService = dbService;
            _analyzerService = analyzerService;
        }

        public async Task LoadPhotosAsync()
        {
            StatusMessage = "Loading photos from database...";
            _allPhotosCache = await _dbService.GetAllPhotosAsync();
            ApplyFilters();
            StatusMessage = $"Loaded {_allPhotosCache.Count} photos.";
        }

        public async Task OpenFolderAsync(string rootFolderPath)
        {
            StatusMessage = $"Scanning {rootFolderPath}...";
            FolderNodes.Clear();
            
            var rootNode = new FolderNode
            {
                FolderName = Path.GetFileName(rootFolderPath) ?? rootFolderPath,
                FolderPath = rootFolderPath,
                IsExpanded = true
            };
            
            await Task.Run(() => BuildFolderTree(rootFolderPath, rootNode));
            
            FolderNodes.Add(rootNode);

            // Load all files from these folders into our cache if they aren't already
            await LoadFilesFromFolderAsync(rootFolderPath);
            
            SelectedFolderNode = rootNode;
            StatusMessage = "Folder opened.";
        }

        private void BuildFolderTree(string path, FolderNode parentNode)
        {
            try
            {
                var dirs = Directory.GetDirectories(path);
                foreach (var dir in dirs)
                {
                    var node = new FolderNode
                    {
                        FolderName = Path.GetFileName(dir),
                        FolderPath = dir
                    };
                    BuildFolderTree(dir, node);
                    
                    // Only add if it or its children have photos, but for simplicity we'll add all for now
                    // Or we could check for files
                    parentNode.Children.Add(node);
                }
            }
            catch (UnauthorizedAccessException) { }
        }

        private async Task LoadFilesFromFolderAsync(string folderPath)
        {
            var files = Directory.GetFiles(folderPath, "*.*", SearchOption.AllDirectories)
                .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".png", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".heic", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".cr2", StringComparison.OrdinalIgnoreCase) ||
                            f.EndsWith(".nef", StringComparison.OrdinalIgnoreCase))
                .ToList();

            var dbPhotos = await _dbService.GetAllPhotosAsync();
            _allPhotosCache = dbPhotos;

            foreach (var file in files)
            {
                if (!_allPhotosCache.Any(p => p.FilePath.Equals(file, StringComparison.OrdinalIgnoreCase)))
                {
                    var fileInfo = new FileInfo(file);
                    _allPhotosCache.Add(new PhotoItem
                    {
                        FilePath = file,
                        FileName = fileInfo.Name,
                        FileSizeBytes = fileInfo.Length,
                        DateTaken = fileInfo.CreationTime,
                        IsAnalyzed = false
                    });
                }
            }
        }

        partial void OnSelectedFolderNodeChanged(FolderNode? value) => ApplyFilters();
        partial void OnShowFavoritesOnlyChanged(bool value) => ApplyFilters();
        partial void OnShowPeopleOnlyChanged(bool value) => ApplyFilters();

        private void ApplyFilters()
        {
            if (SelectedFolderNode == null)
            {
                Photos.Clear();
                return;
            }

            var filtered = _allPhotosCache.AsEnumerable();
            
            // Filter by folder recursively
            filtered = filtered.Where(p => p.FilePath.StartsWith(SelectedFolderNode.FolderPath, StringComparison.OrdinalIgnoreCase));

            if (ShowFavoritesOnly)
            {
                filtered = filtered.Where(p => p.IsFavorite);
            }
            if (ShowPeopleOnly)
            {
                filtered = filtered.Where(p => p.FaceCount > 0);
            }
            
            Photos = new ObservableCollection<PhotoItem>(filtered.OrderBy(p => p.DateTaken));
        }

        [RelayCommand]
        private async Task AnalyzePhotosAsync()
        {
            if (SelectedFolderNode == null) return;

            IsAnalyzing = true;
            var toAnalyze = Photos.Where(p => !p.IsAnalyzed).ToList();
            
            TotalPhotos = toAnalyze.Count;
            AnalyzedPhotos = 0;

            if (TotalPhotos == 0)
            {
                StatusMessage = "All photos in this folder are already analyzed.";
                IsAnalyzing = false;
                return;
            }

            await _analyzerService.InitializeAsync();

            foreach (var photo in toAnalyze)
            {
                StatusMessage = $"Analyzing... {AnalyzedPhotos}/{TotalPhotos}";
                
                var analyzedPhoto = await _analyzerService.AnalyzePhotoAsync(photo.FilePath);
                
                // Copy values
                photo.IsAnalyzed = true;
                photo.SharpnessScore = analyzedPhoto.SharpnessScore;
                photo.FaceCount = analyzedPhoto.FaceCount;
                photo.SceneCategory = analyzedPhoto.SceneCategory;
                
                if (analyzedPhoto.DateTaken != default && analyzedPhoto.DateTaken != photo.DateTaken)
                {
                    photo.DateTaken = analyzedPhoto.DateTaken;
                }

                await _dbService.SavePhotoAsync(photo);
                
                AnalyzedPhotos++;
            }

            StatusMessage = $"Analyzed {TotalPhotos} photos.";
            IsAnalyzing = false;
            ApplyFilters(); // Refresh display
        }

        [RelayCommand]
        private async Task ExportFavoritesAsync(IntPtr hwnd)
        {
            var favorites = _allPhotosCache.Where(p => p.IsFavorite).ToList();
            if (!favorites.Any()) return;
            var folderPicker = new FolderPicker();
            InitializeWithWindow.Initialize(folderPicker, hwnd);
            folderPicker.FileTypeFilter.Add("*");

            var destFolder = await folderPicker.PickSingleFolderAsync();
            if (destFolder == null) return;
            StatusMessage = "Exporting favorites...";
            int count = 0;
            foreach (var photo in favorites)
            {
                var destPath = Path.Combine(destFolder.Path, photo.FileName);
                if (!File.Exists(destPath))


                {
                    File.Copy(photo.FilePath, destPath);
                }
                count++;
            }
            StatusMessage = $"Exported {count} photos to {destFolder.Name}.";
        }


        [RelayCommand]
        private async Task AutoGenerateAlbumAsync()
        {
            // We will hook this up in code-behind to pass the destination folder
        }
        
        public async Task ExecuteAutoGenerateAlbumAsync(string destFolderPath)
        {
            if (!_allPhotosCache.Any()) return;

            StatusMessage = "Generating album pages...";
            
            // 1. Filter out blurry or unanalyzed photos
            var candidates = _allPhotosCache.Where(p => p.IsAnalyzed && p.SharpnessScore > 100).ToList();

            // 2. Sort by date
            candidates = candidates.OrderBy(p => p.DateTaken).ToList();

            // 3. Simple clustering by time (e.g. gap > 4 hours means new event)
            var events = new List<List<PhotoItem>>();
            var currentEvent = new List<PhotoItem>();
            DateTime lastDate = DateTime.MinValue;

            foreach (var photo in candidates)
            {
                if (lastDate == DateTime.MinValue || (photo.DateTaken - lastDate).TotalHours > 4)
                {
                    currentEvent = new List<PhotoItem>();
                    events.Add(currentEvent);
                }
                currentEvent.Add(photo);
                lastDate = photo.DateTaken;
            }

            // 4. Select top photos per event (target ~400 total)
            int targetTotal = 400;
            int selectedTotal = 0;
            var selectedPhotos = new List<PhotoItem>();

            foreach (var ev in events)
            {
                int quota = Math.Max(1, (int)((double)ev.Count / candidates.Count * targetTotal));
                
                var bestInEvent = ev.OrderByDescending(p => p.SharpnessScore + (p.FaceCount * 500))
                                    .Take(quota)
                                    .OrderBy(p => p.DateTaken) 
                                    .ToList();
                
                selectedPhotos.AddRange(bestInEvent);
                selectedTotal += bestInEvent.Count;
            }

            // 5. Group into pages of 3 to 9 photos
            var pages = new List<List<PhotoItem>>();
            var currentPage = new List<PhotoItem>();
            
            foreach (var photo in selectedPhotos)
            {
                currentPage.Add(photo);
                if (currentPage.Count >= 6)
                {
                    pages.Add(currentPage);
                    currentPage = new List<PhotoItem>();
                }
            }
            if (currentPage.Any()) pages.Add(currentPage);

            // 6. Export the pages to folders
            int pageIndex = 1;
            foreach (var page in pages)
            {
                var pageFolder = Path.Combine(destFolderPath, $"Page_{pageIndex:D3}");
                Directory.CreateDirectory(pageFolder);
                
                int photoIndex = 1;
                foreach (var photo in page)
                {
                    var destPath = Path.Combine(pageFolder, $"{pageIndex:D3}_{photoIndex:D2}_{photo.FileName}");
                    if (!File.Exists(destPath))
                    {
                        File.Copy(photo.FilePath, destPath);
                    }
                    photoIndex++;
                }
                pageIndex++;
            }

            StatusMessage = $"Generated {pages.Count} pages with {selectedPhotos.Count} total photos.";
        }
    }
}
