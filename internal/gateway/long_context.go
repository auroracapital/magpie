package gateway

import (
	"bytes"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// A Claude subscription account Anthropic has told needs extra usage for
// long context is held by Claude Code's own check to 200K, where another
// account of the same subscription, with extra usage on, answers 752K
// (sam's pool, 2026-11-12: four of twelve accounts refused a 344K
// conversation, one answered it). The refusal is the account's, not the
// model's: withRoom took it for the model's window and left out every
// other account of it, and the conversation, kept on the account that
// refused it, was refused again every few seconds. What each account
// held, or refused, is remembered here, per account and for a while: a
// conversation past 200K is not asked of an account known to stop at it,
// one that answered past it is asked first, and the refusal itself hands
// the conversation to the next account that may hold it, or to the
// provider's Fallback.

// longContextFloor is the size, in tokens, Claude Code holds an account
// without long context to.
const longContextFloor = 200_000

// longContextKeep is how long what an account held, or refused, is
// believed: extra usage may be turned on, or off, in the meantime.
const longContextKeep = 6 * time.Hour

// ownCheckWords are bridgeFailed's, for Claude Code's own too-long check.
const ownCheckWords = "Claude Code's own check"

var longContext = struct {
	sync.Mutex
	short map[string]time.Time // account (who) → when it refused a conversation as too long for it
	long  map[string]time.Time // account (who) → when it answered one past longContextFloor
}{short: map[string]time.Time{}, long: map[string]time.Time{}}

// shortContextRefusal is whether c's failure, at status with body, for a
// request of about tokens, is Claude Code's own check holding this
// account to its limit, not the model's window: c is a Claude
// subscription account, the conversation is within its model's window
// as far as it is known, and c has not answered as long a one lately
// (an account that holds long context and still refused it has hit the
// model's own limit, which every account has).
func shortContextRefusal(c candidate, status int, body []byte, tokens int) bool {
	if !c.claudeAccount() || status != http.StatusBadRequest {
		return false
	}
	if !bytes.Contains(wordsOf(body), []byte(ownCheckWords)) {
		return false
	}
	if w := windowOf(c.p, c.model); w > 0 && tokens >= w {
		return false
	}
	return !knownLong(c)
}

// noteShortContext remembers that c holds no more than longContextFloor.
func noteShortContext(c candidate) {
	longContext.Lock()
	defer longContext.Unlock()
	longContext.short[c.who()] = time.Now()
	delete(longContext.long, c.who())
}

// noteLongContext remembers that c answered a conversation of tokens,
// when that is past longContextFloor: it holds long context.
func noteLongContext(c candidate, tokens int) {
	if !c.claudeAccount() || tokens < longContextFloor {
		return
	}
	longContext.Lock()
	defer longContext.Unlock()
	longContext.long[c.who()] = time.Now()
	delete(longContext.short, c.who())
}

func knownShort(c candidate) bool { return learnedWithin(longContext.short, c) }
func knownLong(c candidate) bool  { return learnedWithin(longContext.long, c) }

func learnedWithin(m map[string]time.Time, c candidate) bool {
	if !c.claudeAccount() {
		return false
	}
	longContext.Lock()
	defer longContext.Unlock()
	at, ok := m[c.who()]
	if ok && time.Since(at) > longContextKeep {
		delete(m, c.who())
		return false
	}
	return ok
}

// longerThan is, of the candidates left after c refused a request of
// about tokens as past its account's own limit, those that may hold it:
// the other accounts of c's model at c's provider not known to stop at
// longContextFloor, those known to hold more first, then the members
// withRoom leaves (another model, the provider's Fallback).
func longerThan(left []candidate, c candidate, tokens int) []candidate {
	var long, unknown []candidate
	for _, x := range left {
		if x.p.ID != c.p.ID || x.model != c.model || !x.claudeAccount() || x.who() == c.who() {
			continue
		}
		if knownShort(x) || !holds(x, tokens) {
			continue
		}
		if knownLong(x) {
			long = append(long, x)
		} else {
			unknown = append(unknown, x)
		}
	}
	out := append(long, unknown...)
	return append(out, withRoom(left, c, tokens)...)
}

// roomForLong leaves out, of cs, before any is asked, the Claude
// subscription accounts known to stop at longContextFloor when the
// request is past it, and puts the accounts known to hold more before
// the other accounts, in the places those accounts had. When none would
// be left, cs is as it was: the first is asked, and its refusal told.
func roomForLong(cs []candidate, pl planned, from provider.Protocol, body []byte) ([]candidate, planned) {
	if len(cs) < 2 || len(body)/4 < longContextFloor {
		return cs, pl
	}
	if !containsClaudeAccount(cs) {
		return cs, pl
	}
	req, err := parse(from, body)
	if err != nil || estimate(req) < longContextFloor {
		return cs, pl
	}
	parallel := len(pl.order) == len(cs)
	var out []candidate
	var order []Weighed
	var slots, longs, others []int // indices into out of the accounts kept
	for i, c := range cs {
		if knownShort(c) {
			continue
		}
		if c.claudeAccount() {
			slots = append(slots, len(out))
			if knownLong(c) {
				longs = append(longs, len(out))
			} else {
				others = append(others, len(out))
			}
		}
		out = append(out, c)
		if parallel {
			order = append(order, pl.order[i])
		}
	}
	if len(out) == 0 {
		return cs, pl
	}
	if len(longs) > 0 && len(others) > 0 {
		// the accounts known to hold long context take the first of the
		// accounts' places, in the order they had
		picks := append(longs, others...)
		movedC := make([]candidate, len(picks))
		var movedW []Weighed
		for k, j := range picks {
			movedC[k] = out[j]
			if parallel {
				movedW = append(movedW, order[j])
			}
		}
		for k, j := range slots {
			out[j] = movedC[k]
			if parallel {
				order[j] = movedW[k]
			}
		}
	}
	if len(out) < len(cs) {
		log.Printf("claude: a conversation past %dK tokens is not asked of %d account(s) known to hold no more", longContextFloor/1000, len(cs)-len(out))
	}
	if parallel {
		pl.order = order
	}
	return out, pl
}

func containsClaudeAccount(cs []candidate) bool {
	for _, c := range cs {
		if c.claudeAccount() {
			return true
		}
	}
	return false
}
