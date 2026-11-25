import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Official Swift async/await client for JouleDB v1.0.0 (macOS, iOS, visionOS, Linux).
public actor JouleDBClient {
    public static let version = "1.0.0"
    private let baseURL: URL
    private let token: String?

    public init(url: String = "http://127.0.0.1:8080", token: String? = nil) throws {
        let trimmed = url.hasSuffix("/") ? String(url.dropLast()) : url
        guard let parsed = URL(string: trimmed) else {
            throw URLError(.badURL)
        }
        self.baseURL = parsed
        self.token = token
    }

    public func query(_ sql: String) async throws -> [String: Any] {
        try await postJSON(path: "/api/v1/query", body: ["sql": sql])
    }

    public func migrateCertify() async throws -> [String: Any] {
        try await postJSON(path: "/api/v1/migrate/certify", body: [:])
    }

    public func trustAudit() async throws -> [String: Any] {
        try await postJSON(path: "/api/v1/trust/compliance_audit", body: [:])
    }

    public func powerhouseSuite() async throws -> [String: Any] {
        try await postJSON(path: "/api/v1/bench/v1_powerhouse", body: [:])
    }

    private func postJSON(path: String, body: [String: Any]) async throws -> [String: Any] {
        guard let endpoint = URL(string: baseURL.absoluteString + path) else {
            throw URLError(.badURL)
        }
        var req = URLRequest(url: endpoint)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let token = token {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let httpResp = resp as? HTTPURLResponse, httpResp.statusCode >= 400 {
            throw URLError(.badServerResponse)
        }
        let decoded = try JSONSerialization.jsonObject(with: data)
        return (decoded as? [String: Any]) ?? [:]
    }
}
