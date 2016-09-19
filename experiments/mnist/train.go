package main

import (
	"log"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	StepSize  = 0.001
	BatchSize = 16
)

func TrainNetwork(net rnn.Block, training, testing sgd.SampleSet) {
	gradienter := &sgd.RMSProp{
		Gradienter: &seqtoseq.BPTT{
			Block:    net,
			Learner:  net.(sgd.Learner),
			CostFunc: &neuralnet.DotCost{},
		},
		Resiliency: 0.9,
	}
	var epoch int
	var lastBatch sgd.SampleSet
	sgd.SGDMini(gradienter, training, StepSize, BatchSize, func(batch sgd.SampleSet) bool {
		sgd.ShuffleSampleSet(testing)
		validation := totalCost(net, testing.Subset(0, BatchSize))
		if lastBatch == nil {
			log.Printf("epoch %d: validation=%f training=%f", epoch, validation,
				totalCost(net, batch))
		} else {
			log.Printf("epoch %d: validation=%f training=%f last=%f", epoch, validation,
				totalCost(net, batch), totalCost(net, lastBatch))
		}
		epoch++
		lastBatch = batch
		return true
	})
}

func totalCost(net rnn.Block, batch sgd.SampleSet) float64 {
	return seqtoseq.TotalCostBlock(net, BatchSize, batch, &neuralnet.DotCost{})
}
