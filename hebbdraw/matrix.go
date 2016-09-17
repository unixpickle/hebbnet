package hebbdraw

import (
	"image"
	"image/color"
	"math"

	"github.com/llgcode/draw2d/draw2dimg"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	dotRadius  = 5
	dotStroke  = 1
	dotPadding = 3
)

func VisualizeMatrix(m *linalg.Matrix) image.Image {
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)
	for _, x := range m.Data {
		if x < minValue {
			minValue = x
		}
		if x > maxValue {
			maxValue = x
		}
	}

	img := image.NewRGBA(image.Rect(0, 0, dotPadding*(m.Cols+1)+dotRadius*2*m.Cols,
		dotPadding*(m.Rows+1)+dotRadius*2*m.Rows))
	gc := draw2dimg.NewGraphicContext(img)
	gc.SetStrokeColor(color.RGBA{A: 0xff})
	for i := 0; i < m.Rows; i++ {
		y := float64(dotPadding + i*(dotPadding+dotRadius*2))
		for j := 0; j < m.Cols; j++ {
			val := m.Get(i, j)
			if minValue == maxValue {
				val = 0.5
			} else {
				val -= minValue
				val /= maxValue - minValue
			}
			brightness := uint8(0xff - int(0xff*val+0.5))
			x := float64(dotPadding + j*(dotPadding+dotRadius*2))
			gc.SetFillColor(color.Gray{Y: brightness})
			gc.BeginPath()
			gc.ArcTo(x+dotRadius, y+dotRadius, dotRadius, dotRadius, 0, math.Pi*2)
			gc.Close()
			gc.FillStroke()
		}
	}
	return img
}
