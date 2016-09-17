package experiments

import "github.com/unixpickle/num-analysis/linalg"

// Converging uses a heuristic to determine if the cost
// is converging, given the history of cost values.
// The minLen parameter specifies the minimum number of
// cost computations needed to decide that convergence
// is taking place.
func Converging(costs []float64, minLen int) bool {
	if len(costs) < minLen {
		return false
	}
	halfGain := mean(costs[len(costs)/2:3*len(costs)/4]) - mean(costs[3*len(costs)/4:])
	if halfGain < 0 {
		return true
	}
	totalGain := costs[0] - mean(costs[3*len(costs)/4:])
	return halfGain/totalGain < 1e-2
}

func mean(list linalg.Vector) float64 {
	var sum float64
	for _, x := range list {
		sum += x
	}
	return sum / float64(len(list))
}
