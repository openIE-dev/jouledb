// Package jouledb provides the official zero-dependency Go SDK for JouleDB v0.19.0.
package jouledb

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const Version = "0.19.0"

// Client connects to a JouleDB v0.19.0 server over HTTP/JSON.
type Client struct {
	BaseURL    string
	Token      string
	HTTPClient *http.Client
}

// QueryResponse holds columns, rows, and verifiable energy telemetry.
type QueryResponse struct {
	Columns         []string        `json:"columns"`
	Rows            [][]any         `json:"rows"`
	AffectedRows    *uint64         `json:"affected_rows,omitempty"`
	ExecutionTimeMs uint64          `json:"execution_time_ms"`
	EnergyJoules    *float64        `json:"energy_joules,omitempty"`
	DeviceTarget    string          `json:"device_target,omitempty"`
	EnergyReceipt   json.RawMessage `json:"energy_receipt,omitempty"`
}

// NewClient constructs a JouleDB Go client.
func NewClient(baseURL string, token string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Token:   token,
		HTTPClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *Client) doJSON(ctx context.Context, method, path string, reqBody any, out any) error {
	var reader io.Reader
	if reqBody != nil {
		buf, err := json.Marshal(reqBody)
		if err != nil {
			return err
		}
		reader = bytes.NewReader(buf)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseURL+path, reader)
	if err != nil {
		return err
	}
	if reqBody != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("jouledb HTTP %d: %s", resp.StatusCode, string(raw))
	}
	return json.Unmarshal(raw, out)
}

// Query executes a SQL statement and returns rows + energy telemetry.
func (c *Client) Query(ctx context.Context, sql string) (*QueryResponse, error) {
	var out QueryResponse
	err := c.doJSON(ctx, http.MethodPost, "/api/v1/query", map[string]any{"sql": sql}, &out)
	return &out, err
}

// MigrateCertify runs the zero-downtime polyglot migration & shadow-traffic parity suite.
func (c *Client) MigrateCertify(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	err := c.doJSON(ctx, http.MethodPost, "/api/v1/migrate/certify", map[string]any{}, &out)
	return out, err
}

// TrustAudit executes continuous SOC2/HIPAA/GDPR/SEC17a4/FedRAMP-MLS compliance checks.
func (c *Client) TrustAudit(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	err := c.doJSON(ctx, http.MethodPost, "/api/v1/trust/compliance_audit", map[string]any{}, &out)
	return out, err
}

// PowerhouseSuite executes the 6 fast in-process subsystem smoke checks (`jouledb self-check`).
func (c *Client) PowerhouseSuite(ctx context.Context) (map[string]any, error) {
	var out map[string]any
	err := c.doJSON(ctx, http.MethodPost, "/api/v1/bench/v1_powerhouse", map[string]any{}, &out)
	return out, err
}
