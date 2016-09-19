package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/unixpickle/hebbnet/experiments"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	HiddenSize = 40
)

func main() {
	rand.Seed(time.Now().UnixNano())

	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "model_name out_path")
		experiments.PrintModels()
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}

	model, ok := experiments.Models[os.Args[1]]
	if !ok {
		fmt.Fprintln(os.Stderr, "Unknown model:", os.Args[1])
		os.Exit(1)
	}

	var networkBlock rnn.StackedBlock
	modelData, err := ioutil.ReadFile(os.Args[2])
	if err == nil {
		networkBlock, err = rnn.DeserializeStackedBlock(modelData)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
			os.Exit(1)
		}
		log.Println("Loaded model with", countParameters(networkBlock), "parameters.")
	} else {
		networkBlock = createBlock(model)
		log.Println("Created model with", countParameters(networkBlock), "parameters.")
	}

	training := SampleSet(mnist.LoadTrainingDataSet().Samples)
	testing := SampleSet(mnist.LoadTestingDataSet().Samples)

	TrainNetwork(networkBlock, training, testing)

	data, err := networkBlock.Serialize()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to serialize:", err)
		os.Exit(1)
	}
	if err := ioutil.WriteFile(os.Args[2], data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to write network:", err)
		os.Exit(1)
	}
}

func createBlock(m experiments.Model) rnn.StackedBlock {
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: 10,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outNet.Randomize()
	return rnn.StackedBlock{
		m.CreateModel(2, HiddenSize),
		m.CreateModel(HiddenSize, HiddenSize),
		rnn.NewNetworkBlock(outNet, 0),
	}
}

func countParameters(b rnn.StackedBlock) int {
	var count int
	for _, p := range b.Parameters() {
		count += len(p.Vector)
	}
	return count
}
