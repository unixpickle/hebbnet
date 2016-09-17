package main

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"os"
	"path/filepath"

	"github.com/unixpickle/hebbnet/hebbdraw"
	"github.com/unixpickle/seqtasks"
)

const (
	BatchSize  = 10
	BatchCount = 20
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage:", os.Args[0], "output_dir")
		os.Exit(1)
	}

	outDir := os.Args[1]

	m := NewModel()
	task := &seqtasks.MatchMultiTask{
		TypeCount: PunctuationCount,
		MinLen:    1,
		MaxLen:    5,
		CloseProb: 0.3,
	}
	for {
		samples := task.NewSamples(BatchSize * BatchCount)
		m.Train(samples)
		score := task.Score(m, BatchSize, BatchCount)
		log.Println("Score is", score)
		if score == 1 {
			break
		}
	}

	log.Println("Saving visualizations...")

	for i, layer := range m.Layers {
		names := []string{
			fmt.Sprintf("layer%d_plasticities.png", i),
			fmt.Sprintf("layer%d_trace_rates.png", i),
		}
		visualizations := []image.Image{
			hebbdraw.VisualizePlasticities(layer),
			hebbdraw.VisualizeTraceRates(layer),
		}
		for i, name := range names {
			vis := visualizations[i]
			outFile, err := os.Create(filepath.Join(outDir, name))
			if err != nil {
				fmt.Fprintln(os.Stderr, "Failed to open output:", err)
				os.Exit(1)
			}
			err = png.Encode(outFile, vis)
			outFile.Close()
			if err != nil {
				fmt.Fprintln(os.Stderr, "Failed to encode output:", err)
				os.Exit(1)
			}
		}
	}
}
