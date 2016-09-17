package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const RandomRestarts = 2

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "model_name hidden1 ...")
		fmt.Fprintln(os.Stderr, "\nAvailable models:")
		for _, name := range CreatorNames {
			fmt.Fprintln(os.Stderr, " -", name)
		}
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}
	creator, ok := Creators[os.Args[1]]
	if !ok {
		fmt.Fprintln(os.Stderr, "Unknown model:", os.Args[1])
		os.Exit(1)
	}

	var hiddenSizes []int
	for _, sizeStr := range os.Args[2:] {
		size, err := strconv.Atoi(sizeStr)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Invalid hidden size:", sizeStr)
			os.Exit(1)
		}
		hiddenSizes = append(hiddenSizes, size)
	}

	minCapacity := 0
	maxCapacity := 1
	fmt.Println("Trying capacity", maxCapacity)
	for testCapacity(creator, hiddenSizes, maxCapacity) {
		minCapacity = maxCapacity
		maxCapacity *= 2
		fmt.Println("Trying capacity", maxCapacity)
	}

	fmt.Println("Bisecting between", minCapacity, "and", maxCapacity)
	for minCapacity+1 < maxCapacity {
		cap := (minCapacity + maxCapacity) / 2
		fmt.Println("Trying capacity", cap)
		if testCapacity(creator, hiddenSizes, cap) {
			minCapacity = cap
		} else {
			maxCapacity = cap
		}
	}
	fmt.Println("Best capacity is", minCapacity)
}

func testCapacity(c Creator, hidden []int, capacity int) bool {
	for i := 0; i <= RandomRestarts; i++ {
		if testCapacityOnce(c, hidden, capacity) {
			return true
		}
	}
	return false
}

func testCapacityOnce(c Creator, hidden []int, capacity int) bool {
	rand.Seed(time.Now().UnixNano())
	b := createBlock(c, hidden)
	sample := seqtoseq.Sample{
		Inputs:  []linalg.Vector{},
		Outputs: []linalg.Vector{},
	}

	rand.Seed(1337)
	for i := 0; i < capacity; i++ {
		out := []float64{float64(rand.Intn(2))}
		in := []float64{0}
		if i > 0 {
			in = sample.Outputs[len(sample.Outputs)-1]
		}
		sample.Inputs = append(sample.Inputs, in)
		sample.Outputs = append(sample.Outputs, out)
	}

	gradienter := &sgd.RMSProp{
		Gradienter: &seqtoseq.BPTT{
			Block:    b,
			Learner:  b.(sgd.Learner),
			CostFunc: &neuralnet.SigmoidCECost{},
		},
		Resiliency: 0.9,
	}
	samples := sgd.SliceSampleSet{sample}
	var costs []float64
	for !isConverging(costs) {
		sgd.SGD(gradienter, samples, 0.001, 1, 1)

		cost := seqtoseq.TotalCostBlock(b, 1, samples, &neuralnet.SigmoidCECost{})
		costs = append(costs, cost)

		r := &rnn.Runner{Block: b}
		perfect := true
		for i, in := range sample.Inputs {
			out := r.StepTime(in)
			expected := sample.Outputs[i]
			if (expected[0] == 1) != (out[0] > 0) {
				perfect = false
				break
			}
		}
		if perfect {
			return true
		}
	}
	return false
}

func createBlock(c Creator, hidden []int) rnn.Block {
	b := rnn.StackedBlock{}
	for i, size := range hidden {
		inputSize := 1
		if i > 0 {
			inputSize = hidden[i-1]
		}
		b = append(b, c.CreateModel(inputSize, size))
	}
	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  hidden[len(hidden)-1],
			OutputCount: 1,
		},
	}
	outNet.Randomize()
	b = append(b, rnn.NewNetworkBlock(outNet, 0))
	return b
}

func isConverging(costs []float64) bool {
	if len(costs) < 400 {
		return false
	}
	halfGain := mean(costs[len(costs)/2:3*len(costs)/4]) - mean(costs[3*len(costs)/4:])
	if halfGain < 0 {
		return true
	}
	totalGain := costs[0] - mean(costs[3*len(costs)/4:])
	if halfGain/totalGain < 1e-3 {
		return true
	}
	return false
}

func mean(list linalg.Vector) float64 {
	var sum float64
	for _, x := range list {
		sum += x
	}
	return sum / float64(len(list))
}
