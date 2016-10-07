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

type result struct {
	label int
	right bool
}

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
	outChan := make(chan result)

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
	correctMap := map[int]int{}
	totalMap := map[int]int{}
	for res := range outChan {
		total++
		totalMap[res.label]++
		if res.right {
			correct++
			correctMap[res.label]++
		}
		histogram := ""
		for i := 0; i < 10; i++ {
			if i != 0 {
				histogram += " "
			}
			ratio := float64(correctMap[i]) / float64(totalMap[i])
			if totalMap[i] == 0 {
				ratio = 0
			}
			histogram += fmt.Sprintf("%d:%0.2f", i, ratio)
		}
		fmt.Printf("Got %d/%d (%.02f%%) %s\n", correct, total,
			100*float64(correct)/float64(total), histogram)
	}
}

func rateSamples(b rnn.Block, in <-chan seqtoseq.Sample, out chan<- result) {
	for input := range in {
		runner := rnn.Runner{Block: b}
		for _, vec := range input.Inputs[:len(input.Inputs)-1] {
			runner.StepTime(vec)
		}
		outVec := runner.StepTime(input.Inputs[len(input.Inputs)-1])
		label := maxIdx(input.Outputs[len(input.Outputs)-1])
		output := maxIdx(outVec)
		out <- result{
			label: label,
			right: label == output,
		}
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

func maxIdx(vec linalg.Vector) int {
	maxVal := math.Inf(-1)
	maxIdx := 0
	for i, x := range vec {
		if x > maxVal {
			maxVal = x
			maxIdx = i
		}
	}
	return maxIdx
}
