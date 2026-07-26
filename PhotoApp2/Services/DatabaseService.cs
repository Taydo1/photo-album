using Microsoft.Data.Sqlite;
using PhotoApp2.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace PhotoApp2.Services
{
    public class DatabaseService
    {
        private readonly string _dbPath;

        public DatabaseService()
        {
            var folder = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var appFolder = Path.Combine(folder, "PhotoApp2");
            Directory.CreateDirectory(appFolder);
            _dbPath = Path.Combine(appFolder, "photos.db");
        }

        public async Task InitializeAsync()
        {
            using var connection = new SqliteConnection($"Data Source={_dbPath}");
            await connection.OpenAsync();

            var command = connection.CreateCommand();
            command.CommandText = @"
                CREATE TABLE IF NOT EXISTS Photos (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    FilePath TEXT UNIQUE NOT NULL,
                    FileName TEXT NOT NULL,
                    DateTaken TEXT NOT NULL,
                    FileSizeBytes INTEGER NOT NULL,
                    IsAnalyzed INTEGER NOT NULL DEFAULT 0,
                    SharpnessScore REAL NOT NULL DEFAULT 0,
                    FaceCount INTEGER NOT NULL DEFAULT 0,
                    SceneCategory TEXT,
                    IsFavorite INTEGER NOT NULL DEFAULT 0
                );
            ";
            await command.ExecuteNonQueryAsync();
        }

        public async Task SavePhotoAsync(PhotoItem photo)
        {
            using var connection = new SqliteConnection($"Data Source={_dbPath}");
            await connection.OpenAsync();

            var command = connection.CreateCommand();
            command.CommandText = @"
                INSERT INTO Photos (FilePath, FileName, DateTaken, FileSizeBytes, IsAnalyzed, SharpnessScore, FaceCount, SceneCategory, IsFavorite)
                VALUES ($FilePath, $FileName, $DateTaken, $FileSizeBytes, $IsAnalyzed, $SharpnessScore, $FaceCount, $SceneCategory, $IsFavorite)
                ON CONFLICT(FilePath) DO UPDATE SET
                    IsAnalyzed = excluded.IsAnalyzed,
                    SharpnessScore = excluded.SharpnessScore,
                    FaceCount = excluded.FaceCount,
                    SceneCategory = excluded.SceneCategory,
                    IsFavorite = excluded.IsFavorite;
            ";

            command.Parameters.AddWithValue("$FilePath", photo.FilePath);
            command.Parameters.AddWithValue("$FileName", photo.FileName);
            command.Parameters.AddWithValue("$DateTaken", photo.DateTaken.ToString("o"));
            command.Parameters.AddWithValue("$FileSizeBytes", photo.FileSizeBytes);
            command.Parameters.AddWithValue("$IsAnalyzed", photo.IsAnalyzed ? 1 : 0);
            command.Parameters.AddWithValue("$SharpnessScore", photo.SharpnessScore);
            command.Parameters.AddWithValue("$FaceCount", photo.FaceCount);
            command.Parameters.AddWithValue("$SceneCategory", photo.SceneCategory ?? "");
            command.Parameters.AddWithValue("$IsFavorite", photo.IsFavorite ? 1 : 0);

            await command.ExecuteNonQueryAsync();
        }

        public async Task<List<PhotoItem>> GetAllPhotosAsync()
        {
            var photos = new List<PhotoItem>();
            using var connection = new SqliteConnection($"Data Source={_dbPath}");
            await connection.OpenAsync();

            var command = connection.CreateCommand();
            command.CommandText = "SELECT * FROM Photos";

            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                photos.Add(new PhotoItem
                {
                    Id = reader.GetInt32(0),
                    FilePath = reader.GetString(1),
                    FileName = reader.GetString(2),
                    DateTaken = DateTime.Parse(reader.GetString(3)),
                    FileSizeBytes = reader.GetInt64(4),
                    IsAnalyzed = reader.GetInt32(5) == 1,
                    SharpnessScore = reader.GetDouble(6),
                    FaceCount = reader.GetInt32(7),
                    SceneCategory = reader.IsDBNull(8) ? "" : reader.GetString(8),
                    IsFavorite = reader.GetInt32(9) == 1
                });
            }

            return photos;
        }

        public async Task UpdateFavoriteStatusAsync(string filePath, bool isFavorite)
        {
            using var connection = new SqliteConnection($"Data Source={_dbPath}");
            await connection.OpenAsync();

            var command = connection.CreateCommand();
            command.CommandText = "UPDATE Photos SET IsFavorite = $IsFavorite WHERE FilePath = $FilePath";
            command.Parameters.AddWithValue("$IsFavorite", isFavorite ? 1 : 0);
            command.Parameters.AddWithValue("$FilePath", filePath);

            await command.ExecuteNonQueryAsync();
        }
    }
}
