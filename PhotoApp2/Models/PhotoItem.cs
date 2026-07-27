using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;

namespace PhotoApp2.Models
{
    public partial class PhotoItem : ObservableObject
    {
        public int Id { get; set; }
        public string FilePath { get; set; } = string.Empty;
        public string FileName { get; set; } = string.Empty;
        public DateTime DateTaken { get; set; }
        public long FileSizeBytes { get; set; }
        
        public Uri FileUri => new Uri(FilePath);

        // Analysis Results
        public bool IsAnalyzed { get; set; }
        public double SharpnessScore { get; set; }
        public int FaceCount { get; set; }
        public List<string> Tags { get; set; } = new();
        public float[]? VisualFeatureVector { get; set; }

        public string TagsDisplay => Tags != null && Tags.Any() ? string.Join(", ", Tags) : "Untagged";

        [ObservableProperty]
        private bool _isFavorite;

        [ObservableProperty]
        private bool _isSelected;

        [ObservableProperty]
        private string? _thumbnailPath;
    }
}
