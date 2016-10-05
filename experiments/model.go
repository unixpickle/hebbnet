package experiments

import (
	"fmt"
	"os"

	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/weakai/rnn"
)

type Model interface {
	CreateModel(in, out int) rnn.Block
}

type HebbModel struct {
	UseActivation bool
	VariableRate  bool
}

func (h *HebbModel) CreateModel(in, out int) rnn.Block {
	res := hebbnet.NewDenseLayer(in, out, h.VariableRate)
	res.UseActivation = h.UseActivation
	res.InitRates(0.1, 0.3)
	return res
}

type LSTMModel struct{}

func (l *LSTMModel) CreateModel(in, out int) rnn.Block {
	return rnn.NewLSTM(in, out)
}

var ModelNames = []string{"hebbfixed", "hebbvariable", "lstm"}

var Models = map[string]Model{
	"hebbfixed":    &HebbModel{UseActivation: true, VariableRate: false},
	"hebbvariable": &HebbModel{UseActivation: true, VariableRate: true},
	"lstm":         &LSTMModel{},
}

func PrintModels() {
	fmt.Fprintln(os.Stderr, "\nAvailable models:")
	for _, name := range ModelNames {
		fmt.Fprintln(os.Stderr, " -", name)
	}
}
