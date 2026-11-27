using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace OpenIE.JouleDB;

/// <summary>
/// Official async .NET client for JouleDB v0.14.0.
/// </summary>
public sealed class JouleDbClient : IDisposable
{
    public const string SdkVersion = "0.14.0";
    private readonly HttpClient _http;
    private readonly bool _ownsClient;

    public JouleDbClient(string baseUrl = "http://127.0.0.1:8080", string? bearerToken = null, HttpClient? httpClient = null)
    {
        _ownsClient = httpClient is null;
        _http = httpClient ?? new HttpClient();
        _http.BaseAddress = new Uri(baseUrl.TrimEnd('/') + "/");
        if (!string.IsNullOrWhiteSpace(bearerToken))
        {
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearerToken);
        }
    }

    public async Task<JsonDocument> QueryAsync(string sql, CancellationToken ct = default)
        => await PostJsonAsync("api/v1/query", new { sql }, ct).ConfigureAwait(false);

    public async Task<JsonDocument> MigrateCertifyAsync(CancellationToken ct = default)
        => await PostJsonAsync("api/v1/migrate/certify", new { }, ct).ConfigureAwait(false);

    public async Task<JsonDocument> TrustAuditAsync(CancellationToken ct = default)
        => await PostJsonAsync("api/v1/trust/compliance_audit", new { }, ct).ConfigureAwait(false);

    public async Task<JsonDocument> PowerhouseSuiteAsync(CancellationToken ct = default)
        => await PostJsonAsync("api/v1/bench/v1_powerhouse", new { }, ct).ConfigureAwait(false);

    private async Task<JsonDocument> PostJsonAsync<T>(string relativePath, T payload, CancellationToken ct)
    {
        var json = JsonSerializer.Serialize(payload);
        using var content = new StringContent(json, Encoding.UTF8, "application/json");
        using var resp = await _http.PostAsync(relativePath, content, ct).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();
        var stream = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        return await JsonDocument.ParseAsync(stream, cancellationToken: ct).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_ownsClient)
        {
            _http.Dispose();
        }
    }
}
