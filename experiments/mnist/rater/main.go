package main

import (
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "model_file")
		os.Exit(1)
	}

	modelData, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to load model:", err)
		os.Exit(1)
	}
	model, err := rnn.DeserializeStackedBlock(modelData)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to deserialize model:", err)
		os.Exit(1)
	}

	samples := mnist.LoadTestingDataSet()
	ch := make(chan seqtoseq.Sample, 1)
	go func() {
		rand.Seed(time.Now().UnixNano())
		perm := rand.Perm(len(samples.Samples))
		for _, j := range perm {
			ch <- recurrentSample(samples.Samples[j])
		}
		close(ch)
	}()
	outChan := make(chan bool)

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			rateSamples(model, ch, outChan)
			wg.Done()
		}()
	}

	go func() {
		wg.Wait()
		close(outChan)
	}()

	var total, correct int
	for right := range outChan {
		total++
		if right {
			correct++
		}
		fmt.Printf("\rGot %d/%d (%.02f%%)   ", correct, total,
			100*float64(correct)/float64(total))
	}
}

func rateSamples(b rnn.Block, in <-chan seqtoseq.Sample, out chan<- bool) {
	for input := range in {
		runner := rnn.Runner{Block: b}
		for _, vec := range input.Inputs[:len(input.Inputs)-1] {
			runner.StepTime(vec)
		}
		outVec := runner.StepTime(input.Inputs[len(input.Inputs)-1])
		maxVal := math.Inf(-1)
		maxIdx := 0
		for i, x := range outVec {
			if x > maxVal {
				maxVal = x
				maxIdx = i
			}
		}
		out <- (input.Outputs[len(input.Outputs)-1][maxIdx] == 1)
	}
}

func recurrentSample(sample mnist.Sample) seqtoseq.Sample {
	var resSample seqtoseq.Sample
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
