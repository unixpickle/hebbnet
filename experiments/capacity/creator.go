package main

import (
	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/weakai/rnn"
)

type Creator interface {
	CreateModel(in, out int) rnn.Block
}

type HebbCreator struct {
	UseActivation bool
	VariableRate  bool
}

func (h *HebbCreator) CreateModel(in, out int) rnn.Block {
	res := hebbnet.NewDenseLayer(in, out, h.VariableRate)
	res.UseActivation = h.UseActivation
	return res
}

type LSTMCreator struct{}

func (l *LSTMCreator) CreateModel(in, out int) rnn.Block {
	return rnn.NewLSTM(in, out)
}

var CreatorNames = []string{"hebbfixed", "hebbvariable", "lstm"}

var Creators = map[string]Creator{
	"hebbfixed":    &HebbCreator{UseActivation: true, VariableRate: false},
	"hebbvariable": &HebbCreator{UseActivation: true, VariableRate: true},
	"lstm":         &LSTMCreator{},
}
