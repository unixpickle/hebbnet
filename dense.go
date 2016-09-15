package hebbnet

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

const defaultTraceRate = 0.1

func init() {
	var d DenseLayer
	serializer.RegisterTypedDeserializer(d.SerializerType(), DeserializeDenseLayer)
}

// A DenseLayer is a fully-connected recurrent layer with
// Hebbian plasticities as well as standard weights.
type DenseLayer struct {
	InputCount  int
	OutputCount int

	// TraceRate specifies how much the Hebbian trace can
	// change between timesteps, where 0 means no change
	// and 1 means complete change.
	TraceRate *autofunc.Variable

	// Weights stores the weight matrix of the layer in a
	// row-major format.
	// There are InputCount columns and OutputCount rows.
	Weights *autofunc.Variable

	// Biases stores the output biases.
	Biases *autofunc.Variable

	// Plasticities is a matrix containing the plasticity
	// for each connection.
	// It is layed out like Weights in memory.
	Plasticities *autofunc.Variable

	// InitTrace is the initial value for the Hebbian trace.
	InitTrace *autofunc.Variable
}

// DeserializeDenseLayer deserializes a DenseLayer.
func DeserializeDenseLayer(d []byte) (*DenseLayer, error) {
	var res DenseLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// NewDenseLayer creates a DenseLayer with pre-initialized
// (semi-randomized) parameters.
func NewDenseLayer(inCount, outCount int) *DenseLayer {
	weightCount := inCount * outCount
	res := &DenseLayer{
		InputCount:   inCount,
		OutputCount:  outCount,
		TraceRate:    &autofunc.Variable{Vector: []float64{defaultTraceRate}},
		Weights:      &autofunc.Variable{Vector: make(linalg.Vector, weightCount)},
		Biases:       &autofunc.Variable{Vector: make(linalg.Vector, outCount)},
		Plasticities: &autofunc.Variable{Vector: make(linalg.Vector, weightCount)},
		InitTrace:    &autofunc.Variable{Vector: make(linalg.Vector, weightCount)},
	}
	weightStddev := 1 / math.Sqrt(float64(inCount))
	for i := 0; i < weightCount; i++ {
		res.Weights.Vector[i] = rand.NormFloat64() * weightStddev
	}
	return res
}

// Parameters returns the layer's learnable parameters.
func (d *DenseLayer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{
		d.TraceRate,
		d.Weights,
		d.Biases,
		d.Plasticities,
		d.InitTrace,
	}
}

// StartState returns the initial trace.
func (d *DenseLayer) StartState() autofunc.Result {
	return d.InitTrace
}

// StartStateR returns the initial trace.
func (d *DenseLayer) StartStateR(rv autofunc.RVector) autofunc.RResult {
	return autofunc.NewRVariable(d.InitTrace, rv)
}

// SerializerType returns the unique ID used to serialize
// this type with the serializer package.
func (d *DenseLayer) SerializerType() string {
	return "github.com/unixpickle/hebbnet.DenseLayer"
}

// Serialize serializes the layer.
func (d *DenseLayer) Serialize() ([]byte, error) {
	return json.Marshal(d)
}
