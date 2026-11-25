package jouledb

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

func TestJouleDBClientQueryAndPowerhouse(t *testing.T) {
	ctx := context.Background()
	if live := os.Getenv("JOULEDB_URL"); live != "" {
		c := NewClient(live, "")
		res, err := c.Query(ctx, "SELECT 7 * 6 AS product")
		if err != nil {
			t.Fatalf("live query failed: %v", err)
		}
		if len(res.Rows) != 1 {
			t.Fatalf("expected 1 row, got %d", len(res.Rows))
		}
		ph, err := c.PowerhouseSuite(ctx)
		if err != nil {
			t.Fatalf("live powerhouse failed: %v", err)
		}
		if ok, _ := ph["all_36_pillars_passed"].(bool); !ok {
			t.Fatalf("expected all_36_pillars_passed=true, got %v", ph["all_36_pillars_passed"])
		}
		return
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/api/v1/query" {
			_, _ = w.Write([]byte(`{"columns":["product"],"rows":[[42]],"execution_time_ms":1}`))
			return
		}
		_, _ = w.Write([]byte(`{"all_36_pillars_passed":true,"total_certified_pillars":36}`))
	}))
	defer ts.Close()

	c := NewClient(ts.URL, "test-token")
	res, err := c.Query(ctx, "SELECT 42 AS product")
	if err != nil {
		t.Fatalf("query error: %v", err)
	}
	if len(res.Rows) != 1 {
		t.Fatalf("expected 1 row")
	}
	ph, err := c.PowerhouseSuite(ctx)
	if err != nil {
		t.Fatalf("powerhouse error: %v", err)
	}
	if ok, _ := ph["all_36_pillars_passed"].(bool); !ok {
		t.Fatalf("expected all_36_pillars_passed=true")
	}
}
