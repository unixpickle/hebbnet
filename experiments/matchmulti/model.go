package main

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	PunctuationCount = 2
	InSize           = 2*PunctuationCount + 1
	OutSize          = PunctuationCount + 1
	HiddenSize       = 20
	LayerCount       = 3
	StepSize         = 0.01
)

// A Model is a seqtasks.Model which uses a stacked block.
type Model struct {
	Layers []*hebbnet.DenseLayer
	Block  rnn.StackedBlock

	gradienter sgd.Gradienter
}

func NewModel() *Model {
	res := &Model{}
	outNet := neuralnet.Network{
		&neuralnet.LogSoftmaxLayer{},
	}
	outNet.Randomize()
	outBlock := rnn.NewNetworkBlock(outNet, 0)
	for i := 0; i < LayerCount; i++ {
		var layer *hebbnet.DenseLayer
		inSize := InSize
		if i > 0 {
			inSize = HiddenSize
		}
		outSize := HiddenSize
		if i+1 == LayerCount {
			outSize = OutSize
		}
		layer = hebbnet.NewDenseLayer(inSize, outSize, true)
		layer.UseActivation = true
		res.Block = append(res.Block, layer)
		res.Layers = append(res.Layers, layer)
	}
	res.Block = append(res.Block, outBlock)
	return res
}

func (s *Model) Train(samples sgd.SampleSet) {
	if s.gradienter == nil {
		s.gradienter = &sgd.RMSProp{
			Gradienter: &seqtoseq.BPTT{
				Block:    s.Block,
				Learner:  s.Block,
				CostFunc: &neuralnet.DotCost{},
			},
			Resiliency: 0.9,
		}
	}
	sgd.SGD(s.gradienter, samples, StepSize, 1, BatchSize)
}

func (s *Model) Run(inputs [][]linalg.Vector) [][]linalg.Vector {
	runner := rnn.Runner{Block: s.Block}
	out := runner.RunAll(inputs)
	for _, seq := range out {
		for i, x := range seq {
			in := &autofunc.Variable{Vector: x}
			seq[i] = autofunc.Exp{}.Apply(in).Output()
		}
	}
	return out
}
