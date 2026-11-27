package dev.openie.jouledb;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Official zero-dependency Java 17+ SDK for JouleDB v0.15.0.
 */
public final class JouleDbClient {
    public static final String VERSION = "0.15.0";
    private final String baseUrl;
    private final String bearerToken;
    private final HttpClient http;

    public JouleDbClient(String baseUrl, String bearerToken) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.bearerToken = bearerToken;
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(15))
                .build();
    }

    public JouleDbClient(String baseUrl) {
        this(baseUrl, null);
    }

    public String queryRawJson(String sql) throws IOException, InterruptedException {
        String escaped = sql.replace("\\", "\\\\").replace("\"", "\\\"");
        return postJson("/api/v1/query", "{\"sql\":\"" + escaped + "\"}");
    }

    public String migrateCertifyRawJson() throws IOException, InterruptedException {
        return postJson("/api/v1/migrate/certify", "{}");
    }

    public String trustAuditRawJson() throws IOException, InterruptedException {
        return postJson("/api/v1/trust/compliance_audit", "{}");
    }

    public String powerhouseSuiteRawJson() throws IOException, InterruptedException {
        return postJson("/api/v1/bench/v1_powerhouse", "{}");
    }

    private String postJson(String path, String jsonBody) throws IOException, InterruptedException {
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(this.baseUrl + path))
                .timeout(Duration.ofSeconds(60))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(jsonBody));
        if (this.bearerToken != null && !this.bearerToken.isEmpty()) {
            b.header("Authorization", "Bearer " + this.bearerToken);
        }
        HttpResponse<String> resp = this.http.send(b.build(), HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IOException("JouleDB HTTP " + resp.statusCode() + ": " + resp.body());
        }
        return resp.body();
    }
}
