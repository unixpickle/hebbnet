package main

import (
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

type SampleSet []mnist.Sample

func (s SampleSet) Len() int {
	return len(s)
}

func (s SampleSet) GetSample(i int) interface{} {
	var resSample seqtoseq.Sample
	sample := s[i]
	for _, x := range sample.Intensities {
		resSample.Inputs = append(resSample.Inputs, []float64{x, 0})
		resSample.Outputs = append(resSample.Outputs, make(linalg.Vector, 10))
	}
	resSample.Inputs = append(resSample.Inputs, []float64{0, 1})
	outVec := make(linalg.Vector, 10)
	outVec[sample.Label] = 1
	resSample.Outputs = append(resSample.Outputs, outVec)
	return resSample
}

func (s SampleSet) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

func (s SampleSet) Copy() sgd.SampleSet {
	res := make(SampleSet, len(s))
	copy(res, s)
	return res
}

func (s SampleSet) Subset(start, end int) sgd.SampleSet {
	return s[start:end]
}
