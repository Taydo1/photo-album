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
                    IsFavorite INTEGER NOT NULL DEFAULT 0,
                    Keywords TEXT NOT NULL DEFAULT '',
                    PrimaryKind TEXT NOT NULL DEFAULT ''
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
                INSERT INTO Photos (FilePath, FileName, DateTaken, FileSizeBytes, IsAnalyzed, SharpnessScore, FaceCount, SceneCategory, IsFavorite, Keywords, PrimaryKind)
                VALUES ($FilePath, $FileName, $DateTaken, $FileSizeBytes, $IsAnalyzed, $SharpnessScore, $FaceCount, $SceneCategory, $IsFavorite, $Keywords, $PrimaryKind)
                ON CONFLICT(FilePath) DO UPDATE SET
                    IsAnalyzed = excluded.IsAnalyzed,
                    SharpnessScore = excluded.SharpnessScore,
                    FaceCount = excluded.FaceCount,
                    SceneCategory = excluded.SceneCategory,
                    IsFavorite = excluded.IsFavorite,
                    Keywords = excluded.Keywords,
                    PrimaryKind = excluded.PrimaryKind;
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
            command.Parameters.AddWithValue("$Keywords", photo.Keywords ?? "");
            command.Parameters.AddWithValue("$PrimaryKind", photo.PrimaryKind ?? "");

            await command.ExecuteNonQueryAsync();
        }

        public async Task SavePhotosAsync(IEnumerable<PhotoItem> photos)
        {
            if (photos == null || !photos.Any()) return;

            using var connection = new SqliteConnection($"Data Source={_dbPath}");
            await connection.OpenAsync();
            using var transaction = connection.BeginTransaction();

            var command = connection.CreateCommand();
            command.Transaction = transaction;
            command.CommandText = @"
                INSERT INTO Photos (FilePath, FileName, DateTaken, FileSizeBytes, IsAnalyzed, SharpnessScore, FaceCount, SceneCategory, IsFavorite, Keywords, PrimaryKind)
                VALUES ($FilePath, $FileName, $DateTaken, $FileSizeBytes, $IsAnalyzed, $SharpnessScore, $FaceCount, $SceneCategory, $IsFavorite, $Keywords, $PrimaryKind)
                ON CONFLICT(FilePath) DO UPDATE SET
                    IsAnalyzed = excluded.IsAnalyzed,
                    SharpnessScore = excluded.SharpnessScore,
                    FaceCount = excluded.FaceCount,
                    SceneCategory = excluded.SceneCategory,
                    IsFavorite = excluded.IsFavorite,
                    Keywords = excluded.Keywords,
                    PrimaryKind = excluded.PrimaryKind;
            ";

            var filePathParam = command.Parameters.Add("$FilePath", SqliteType.Text);
            var fileNameParam = command.Parameters.Add("$FileName", SqliteType.Text);
            var dateTakenParam = command.Parameters.Add("$DateTaken", SqliteType.Text);
            var fileSizeBytesParam = command.Parameters.Add("$FileSizeBytes", SqliteType.Integer);
            var isAnalyzedParam = command.Parameters.Add("$IsAnalyzed", SqliteType.Integer);
            var sharpnessScoreParam = command.Parameters.Add("$SharpnessScore", SqliteType.Real);
            var faceCountParam = command.Parameters.Add("$FaceCount", SqliteType.Integer);
            var sceneCategoryParam = command.Parameters.Add("$SceneCategory", SqliteType.Text);
            var isFavoriteParam = command.Parameters.Add("$IsFavorite", SqliteType.Integer);
            var keywordsParam = command.Parameters.Add("$Keywords", SqliteType.Text);
            var primaryKindParam = command.Parameters.Add("$PrimaryKind", SqliteType.Text);

            foreach (var photo in photos)
            {
                filePathParam.Value = photo.FilePath;
                fileNameParam.Value = photo.FileName;
                dateTakenParam.Value = photo.DateTaken.ToString("o");
                fileSizeBytesParam.Value = photo.FileSizeBytes;
                isAnalyzedParam.Value = photo.IsAnalyzed ? 1 : 0;
                sharpnessScoreParam.Value = photo.SharpnessScore;
                faceCountParam.Value = photo.FaceCount;
                sceneCategoryParam.Value = photo.SceneCategory ?? "";
                isFavoriteParam.Value = photo.IsFavorite ? 1 : 0;
                keywordsParam.Value = photo.Keywords ?? "";
                primaryKindParam.Value = photo.PrimaryKind ?? "";

                await command.ExecuteNonQueryAsync();
            }

            await transaction.CommitAsync();
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
                    IsFavorite = reader.GetInt32(9) == 1,
                    Keywords = reader.FieldCount > 10 && !reader.IsDBNull(10) ? reader.GetString(10) : "",
                    PrimaryKind = reader.FieldCount > 11 && !reader.IsDBNull(11) ? reader.GetString(11) : ""
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
