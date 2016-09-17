package hebbdraw

import (
	"image"

	"github.com/unixpickle/hebbnet"
	"github.com/unixpickle/num-analysis/linalg"
)

func VisualizePlasticities(h *hebbnet.DenseLayer) image.Image {
	return VisualizeMatrix(&linalg.Matrix{
		Data: h.Plasticities.Vector,
		Rows: h.OutputCount,
		Cols: h.InputCount,
	})
}

func VisualizeTraceRates(h *hebbnet.DenseLayer) image.Image {
	if len(h.TraceRate.Vector) == 1 {
		return VisualizeMatrix(&linalg.Matrix{
			Data: h.TraceRate.Vector,
			Rows: 1,
			Cols: 1,
		})
	} else {
		return VisualizeMatrix(&linalg.Matrix{
			Data: h.TraceRate.Vector,
			Rows: h.OutputCount,
			Cols: h.InputCount,
		})
	}
}
