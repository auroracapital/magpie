package gateway

import (
	"testing"

	"github.com/yetone/magpie/internal/settings"
)

// The idle runs kept follow the user's claudeRunsMost, less idleBusy for
// runs answering, never fewer than idleMost nor more than claudeRunsMost;
// on auto they stay idleMost.
func TestIdleCapFollowsClaudeRunsMost(t *testing.T) {
	fresh(t)
	for _, c := range []struct{ most, want int }{
		{0, idleMost}, // auto
		{32, 28},
		{10, 6},
		{8, idleMost},
		{4, 4},
		{1, 1},
	} {
		s := settings.Load()
		s.ClaudeRunsMost = c.most
		if err := settings.Save(s); err != nil {
			t.Fatal(err)
		}
		if got := idleCap(); got != c.want {
			t.Errorf("claudeRunsMost %d: idle cap %d, want %d", c.most, got, c.want)
		}
	}
}
