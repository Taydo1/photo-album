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
                        DateTaken = fileInfo.CreationTime < fileInfo.LastWriteTime ? fileInfo.CreationTime : fileInfo.LastWriteTime,
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
            var toAnalyze = Photos.Where(p => !p.IsAnalyzed || p.Tags == null || !p.Tags.Any()).ToList();
            
            TotalPhotos = toAnalyze.Count;
            AnalyzedPhotos = 0;

            if (TotalPhotos == 0)
            {
                StatusMessage = "All photos in this folder are already analyzed.";
                IsAnalyzing = false;
                return;
            }

            await _analyzerService.InitializeAsync();
            await _dbService.InitializeAsync();

            var progress = new Progress<int>(count =>
            {
                AnalyzedPhotos = count;
                StatusMessage = $"Analyzing... {count}/{TotalPhotos}";
            });
            var reporter = (IProgress<int>)progress;
            int completedCount = 0;

            // Process in rolling batches of 100 photos as requested
            const int batchSize = 100;
            var chunks = toAnalyze.Chunk(batchSize);

            foreach (var chunk in chunks)
            {
                await Task.Run(async () =>
                {
                    await Parallel.ForEachAsync(chunk, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, async (photo, ct) =>
                    {
                        var analyzedPhoto = await _analyzerService.AnalyzePhotoAsync(photo);

                        // Copy values
                        photo.IsAnalyzed = true;
                        photo.SharpnessScore = analyzedPhoto.SharpnessScore;
                        photo.FaceCount = analyzedPhoto.FaceCount;
                        photo.Tags = analyzedPhoto.Tags;
                        photo.VisualFeatureVector = analyzedPhoto.VisualFeatureVector;

                        if (analyzedPhoto.DateTaken != default && analyzedPhoto.DateTaken != photo.DateTaken)
                        {
                            photo.DateTaken = analyzedPhoto.DateTaken;
                        }

                        var current = System.Threading.Interlocked.Increment(ref completedCount);
                        reporter.Report(current);
                    });
                });

                // Save completed batch of up to 100 photos in a single high-speed database transaction
                StatusMessage = $"Saving batch of {chunk.Length} photos to database...";
                await _dbService.SavePhotosAsync(chunk);
            }

            StatusMessage = $"Analyzed {TotalPhotos} photos.";
            IsAnalyzing = false;
            ApplyFilters(); // Refresh display
        }

        [RelayCommand]
        private async Task ClearAllTagsAsync()
        {
            if (Photos == null || !Photos.Any()) return;

            StatusMessage = "Clearing tags for all photos...";
            foreach (var photo in Photos)
            {
                photo.Tags = new List<string>();
                photo.IsAnalyzed = false;
            }

            await _dbService.SavePhotosAsync(Photos);
            StatusMessage = $"Cleared tags for {Photos.Count} photos. Click 'Analyze Photos' to re-tag.";
            ApplyFilters();
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

            var generator = new AlbumGeneratorService();
            var progress = new Progress<string>(msg => StatusMessage = msg);

            var pages = await generator.GenerateAlbumAsync(_allPhotosCache, destFolderPath, progress);

            StatusMessage = $"Generated {pages.Count} album pages in {Path.GetFileName(destFolderPath)}.";
        }
    }
}
