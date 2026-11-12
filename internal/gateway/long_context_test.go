package gateway

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
	"github.com/yetone/magpie/internal/testenv"
)

// freshLongContext empties what accounts were learned to hold, for a test
// alone.
func freshLongContext(t *testing.T) {
	t.Helper()
	reset := func() {
		longContext.Lock()
		longContext.short = map[string]time.Time{}
		longContext.long = map[string]time.Time{}
		longContext.Unlock()
	}
	reset()
	t.Cleanup(reset)
}

func claudeAt(user string) candidate {
	return candidate{
		p:     provider.Provider{ID: "claude", Account: &provider.Account{Agent: "claude", User: user}, Contexts: map[string]int{"claude-opus-5-5": 1_000_000}},
		model: "claude-opus-5-5",
		rest:  "claude",
	}
}

func cursorFallback() candidate {
	return candidate{
		p:     provider.Provider{ID: "cursor", Contexts: map[string]int{"claude-opus-5-5@1m": 1_000_000}},
		model: "claude-opus-5-5@1m",
		rest:  "cursor",
	}
}

const ownCheckRefusal = `{"type":"error","error":{"type":"invalid_request_error","message":"Claude Code: Prompt is too long (Claude Code's own check, before Anthropic was asked: it holds a conversation to its own limit for the account, 200K once Anthropic has said the account needs extra usage for long context. Compact the conversation, or turn on extra usage for the Claude account.)"}}`

// 1. A Claude account's own 200K check is remembered of that account
// alone, for longContextKeep; the model's own window, an account known to
// hold long context, and another vendor's too-long are not it.
func TestShortContextRememberedPerAccount(t *testing.T) {
	freshLongContext(t)
	a, b := claudeAt("a@example.com"), claudeAt("b@example.com")
	if !shortContextRefusal(a, 400, []byte(ownCheckRefusal), 344_000) {
		t.Fatal("Claude Code's own check not read as the account's limit")
	}
	if shortContextRefusal(a, 400, []byte(ownCheckRefusal), 1_200_000) {
		t.Fatal("a conversation past the model's window read as the account's limit")
	}
	if shortContextRefusal(cursorFallback(), 400, []byte(ownCheckRefusal), 344_000) {
		t.Fatal("a provider without Claude accounts read as one")
	}
	if shortContextRefusal(a, 400, []byte(`{"error":{"message":"prompt is too long: 213462 tokens > 200000 maximum"}}`), 213_000) {
		t.Fatal("Anthropic's own too-long read as Claude Code's check")
	}
	noteShortContext(a)
	if !knownShort(a) || knownShort(b) {
		t.Fatalf("short learned of the wrong account: a %v, b %v", knownShort(a), knownShort(b))
	}
	noteLongContext(b, 150_000)
	if knownLong(b) {
		t.Fatal("an answer within 200K taken for long context")
	}
	noteLongContext(b, 752_000)
	if !knownLong(b) || knownShort(b) {
		t.Fatal("an answer past 200K not remembered as long context")
	}
	if shortContextRefusal(b, 400, []byte(ownCheckRefusal), 344_000) {
		t.Fatal("an account known to hold long context read as short")
	}
	// believed for longContextKeep, then forgotten
	longContext.Lock()
	longContext.short[a.who()] = time.Now().Add(-longContextKeep - time.Minute)
	longContext.Unlock()
	if knownShort(a) {
		t.Fatal("a refusal past longContextKeep still believed")
	}
	// answering long clears short, refusing clears long
	noteShortContext(b)
	if knownLong(b) || !knownShort(b) {
		t.Fatal("a new refusal didn't replace what was learned")
	}
}

func pastBody(t *testing.T, tokens int) []byte {
	t.Helper()
	body := []byte(`{"model":"claude/claude-opus-5-5","max_tokens":100,"messages":[{"role":"user","content":"` + strings.Repeat("a", tokens*4) + `"}]}`)
	req, err := parse(provider.Anthropic, body)
	if err != nil || estimate(req) < tokens {
		t.Fatalf("body estimated short: %v", err)
	}
	return body
}

func seatsOf(cs []candidate) []string {
	var out []string
	for _, c := range cs {
		out = append(out, c.who())
	}
	return out
}

// 2. Before any is asked, a conversation past 200K skips the accounts
// known to stop at it and goes first to those known to hold more; with
// none of the accounts left, the provider's Fallback is next; with
// nobody left at all, the list is as it was; and under 200K nothing moves.
func TestLongConversationPicksAccountsThatHoldIt(t *testing.T) {
	freshLongContext(t)
	a, b, c, d := claudeAt("a@example.com"), claudeAt("b@example.com"), claudeAt("c@example.com"), claudeAt("d@example.com")
	fb := cursorFallback()
	noteShortContext(a)
	noteShortContext(c)
	noteLongContext(d, 752_000)
	cs := []candidate{a, b, c, d, fb}
	pl := planned{order: make([]Weighed, len(cs))}
	for i := range pl.order {
		pl.order[i].Who = cs[i].who()
	}

	got, gotPl := roomForLong(slices.Clone(cs), pl, provider.Anthropic, pastBody(t, 300_000))
	want := []string{d.who(), b.who(), fb.who()}
	if !slices.Equal(seatsOf(got), want) {
		t.Fatalf("asked %v, want %v", seatsOf(got), want)
	}
	for i, w := range gotPl.order {
		if w.Who != got[i].who() {
			t.Fatalf("the plan's order no longer matches the candidates: %d %q vs %q", i, w.Who, got[i].who())
		}
	}

	// under 200K every account is asked, in its own order
	small := []byte(`{"model":"claude/claude-opus-5-5","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`)
	if got, _ := roomForLong(slices.Clone(cs), pl, provider.Anthropic, small); !slices.Equal(seatsOf(got), seatsOf(cs)) {
		t.Fatalf("a short conversation moved: %v", seatsOf(got))
	}

	// every account short: the Fallback is what is asked
	noteShortContext(b)
	noteShortContext(d)
	if got, _ := roomForLong(slices.Clone(cs), pl, provider.Anthropic, pastBody(t, 300_000)); !slices.Equal(seatsOf(got), []string{fb.who()}) {
		t.Fatalf("with every account short, asked %v, want the Fallback", seatsOf(got))
	}
	// and no Fallback either: the list is kept, for the refusal to be told
	if got, _ := roomForLong([]candidate{a, b}, planned{}, provider.Anthropic, pastBody(t, 300_000)); len(got) != 2 {
		t.Fatalf("with nobody left, asked %v", seatsOf(got))
	}

	// after a refusal: the other accounts not known short, then the Fallback
	freshLongContext(t)
	noteShortContext(c)
	noteLongContext(d, 752_000)
	if got := longerThan([]candidate{b, c, d, fb}, a, 344_000); !slices.Equal(seatsOf(got), []string{d.who(), b.who(), fb.who()}) {
		t.Fatalf("after a's refusal, left %v", seatsOf(got))
	}
	// nobody after it: none, and the refusal is told
	if got := longerThan([]candidate{c}, a, 344_000); len(got) != 0 {
		t.Fatalf("after a's refusal with only a short account left, left %v", seatsOf(got))
	}
}

// claudeShortOn has the fake Claude Code turn every conversation away with
// its own too-long check on the accounts whose token matches toks (a case
// pattern), and answer on
// the others.
func claudeShortOn(t *testing.T, calls, toks string) {
	t.Helper()
	script := `#!/bin/sh
case "$1" in auth) exit 1;; esac
creds="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/.credentials.json"
while read -r line; do
  tok=$(grep -o 'tok-[a-z]*' "$creds" | head -1)
  echo "$tok" >> '` + calls + `'
  case "$tok" in ` + toks + `)
    echo '{"type":"result","subtype":"success","is_error":true,"result":"Prompt is too long","session_id":"s"}' ;;
  *)
    echo '{"type":"stream_event","event":{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":1}}}}'
    echo '{"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"long account"}}}'
    echo '{"type":"stream_event","event":{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}}'
    echo '{"type":"stream_event","event":{"type":"message_stop"}}'
    echo '{"type":"result","is_error":false,"result":""}' ;;
  esac
done
`
	binary := strings.Split(os.Getenv("PATH"), string(os.PathListSeparator))[0]
	testenv.Program(t, filepath.Join(binary, "claude"), script)
}

// 3. An account's own 200K check is not handed to the client while
// another account may hold the conversation: the next is asked, a new
// Claude Code told the conversation, and it answers; the account that
// refused is remembered as short. With no other account to ask, the 400
// that says to compact is kept.
func TestShortAccountRefusalAsksTheNext(t *testing.T) {
	freshLongContext(t)
	claudeMadeFirst(t, false)
	s := New()
	t.Cleanup(s.subscription.abortAll)
	calls := filepath.Join(t.TempDir(), "calls")
	claudeShortOn(t, calls, "tok-a")
	rec := askClaudeModel(s, "first")
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "long account") {
		t.Fatalf("the refusal handed over, not asked of the next account: %d %s", rec.Code, rec.Body.String())
	}
	if b, _ := os.ReadFile(calls); string(b) != "tok-a\ntok-b\n" {
		t.Fatalf("accounts asked: %q", b)
	}
	routes := s.Trace(context.Background(), 0, 0).Routes
	if len(routes) != 1 || len(routes[0].Tries) != 2 || routes[0].Tries[0].Fail != failOverflow || routes[0].Tries[0].Rest != nil {
		t.Fatalf("tries: %+v", routes)
	}
	if !knownShort(candidate{p: provider.Provider{ID: "claude", Account: &provider.Account{Agent: "claude", User: "a@example.com"}}}) {
		t.Fatal("the account that refused isn't remembered as short")
	}
}

func TestShortAccountRefusalWithNobodyLeftStaysA400(t *testing.T) {
	freshLongContext(t)
	claudeMadeFirst(t, false)
	s := New()
	t.Cleanup(s.subscription.abortAll)
	calls := filepath.Join(t.TempDir(), "calls")
	claudeShortOn(t, calls, "tok-a|tok-b")
	rec := askClaudeModel(s, "first")
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "own check") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if b, _ := os.ReadFile(calls); string(b) != "tok-a\ntok-b\n" {
		t.Fatalf("accounts asked: %q", b)
	}
}
