package hebbnet

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
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

	// If UseActivation is true, then outputs of the layer
	// are fed into hyperbolic tangent and the tangents
	// are used to compute the Hebbian trace.
	UseActivation bool
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

// StateSize returns the size of the state vectors.
func (d *DenseLayer) StateSize() int {
	return d.InputCount * d.OutputCount
}

// StartState returns the initial trace.
func (d *DenseLayer) StartState() autofunc.Result {
	return d.InitTrace
}

// StartStateR returns the initial trace.
func (d *DenseLayer) StartStateR(rv autofunc.RVector) autofunc.RResult {
	return autofunc.NewRVariable(d.InitTrace, rv)
}

// Batch applies the layer to a set of inputs.
func (d *DenseLayer) Batch(in *rnn.BlockInput) rnn.BlockOutput {
	res := &denseLayerOutput{}
	for i := 0; i < len(in.Inputs); i++ {
		newState, out := d.timestep(in.States[i], in.Inputs[i])
		res.StateResults = append(res.StateResults, newState)
		res.OutResults = append(res.OutResults, out)
		res.StateVecs = append(res.StateVecs, newState.Output())
		res.OutVecs = append(res.OutVecs, out.Output())
	}
	return res
}

// BatchR applies the layer to a set of inputs.
func (d *DenseLayer) BatchR(rv autofunc.RVector, in *rnn.BlockRInput) rnn.BlockROutput {
	res := &denseLayerROutput{}
	for i := 0; i < len(in.Inputs); i++ {
		newState, out := d.timestepR(rv, in.States[i], in.Inputs[i])
		res.StateResults = append(res.StateResults, newState)
		res.OutResults = append(res.OutResults, out)
		res.StateVecs = append(res.StateVecs, newState.Output())
		res.OutVecs = append(res.OutVecs, out.Output())
		res.RStateVecs = append(res.RStateVecs, newState.ROutput())
		res.ROutVecs = append(res.ROutVecs, out.ROutput())
	}
	return res
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

func (d *DenseLayer) timestep(state, in autofunc.Result) (newState, out autofunc.Result) {
	weightTran := autofunc.LinTran{
		Data: d.Weights,
		Rows: d.OutputCount,
		Cols: d.InputCount,
	}
	appliedWeights := weightTran.Apply(in)
	appliedHebb := autofunc.MatMulVec(autofunc.Mul(d.Plasticities, state),
		d.OutputCount, d.InputCount, in)
	out = autofunc.Add(d.Biases, autofunc.Add(appliedWeights, appliedHebb))
	if d.UseActivation {
		out = neuralnet.HyperbolicTangent{}.Apply(out)
	}

	keepRate := autofunc.AddScaler(autofunc.Scale(d.TraceRate, -1), 1)
	newState = autofunc.Add(autofunc.ScaleFirst(state, keepRate),
		autofunc.ScaleFirst(autofunc.OuterProduct(out, in), d.TraceRate))

	return
}

func (d *DenseLayer) timestepR(rv autofunc.RVector, state,
	in autofunc.RResult) (newState, out autofunc.RResult) {
	weightTran := autofunc.LinTran{
		Data: d.Weights,
		Rows: d.OutputCount,
		Cols: d.InputCount,
	}
	appliedWeights := weightTran.ApplyR(rv, in)
	plasticState := autofunc.MulR(autofunc.NewRVariable(d.Plasticities, rv), state)
	appliedHebb := autofunc.MatMulVecR(plasticState, d.OutputCount, d.InputCount, in)
	out = autofunc.AddR(autofunc.NewRVariable(d.Biases, rv),
		autofunc.AddR(appliedWeights, appliedHebb))
	if d.UseActivation {
		out = neuralnet.HyperbolicTangent{}.ApplyR(rv, out)
	}

	traceRate := autofunc.NewRVariable(d.TraceRate, rv)
	keepRate := autofunc.AddScalerR(autofunc.ScaleR(traceRate, -1), 1)
	newState = autofunc.AddR(autofunc.ScaleFirstR(state, keepRate),
		autofunc.ScaleFirstR(autofunc.OuterProductR(out, in), traceRate))

	return
}

type denseLayerOutput struct {
	OutVecs      []linalg.Vector
	StateVecs    []linalg.Vector
	OutResults   []autofunc.Result
	StateResults []autofunc.Result
}

func (d *denseLayerOutput) Outputs() []linalg.Vector {
	return d.OutVecs
}

func (d *denseLayerOutput) States() []linalg.Vector {
	return d.StateVecs
}

func (d *denseLayerOutput) Gradient(u *rnn.UpstreamGradient, g autofunc.Gradient) {
	if len(d.StateVecs) == 0 {
		return
	}
	tempStateUpstream := make(linalg.Vector, len(d.StateVecs[0]))
	tempOutUpstream := make(linalg.Vector, len(d.OutVecs[0]))
	for i := 0; i < len(d.OutVecs); i++ {
		if u.States != nil {
			copy(tempStateUpstream, u.States[i])
		}
		if u.Outputs != nil {
			copy(tempOutUpstream, u.Outputs[i])
		}
		d.OutResults[i].PropagateGradient(tempOutUpstream, g)
		d.StateResults[i].PropagateGradient(tempStateUpstream, g)
	}
}

type denseLayerROutput struct {
	OutVecs      []linalg.Vector
	StateVecs    []linalg.Vector
	ROutVecs     []linalg.Vector
	RStateVecs   []linalg.Vector
	OutResults   []autofunc.RResult
	StateResults []autofunc.RResult
}

func (d *denseLayerROutput) Outputs() []linalg.Vector {
	return d.OutVecs
}

func (d *denseLayerROutput) ROutputs() []linalg.Vector {
	return d.ROutVecs
}

func (d *denseLayerROutput) States() []linalg.Vector {
	return d.StateVecs
}

func (d *denseLayerROutput) RStates() []linalg.Vector {
	return d.RStateVecs
}

func (d *denseLayerROutput) RGradient(u *rnn.UpstreamRGradient, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if len(d.StateVecs) == 0 {
		return
	}
	tempStateUpstream := make(linalg.Vector, len(d.StateVecs[0]))
	tempStateUpstreamR := make(linalg.Vector, len(d.StateVecs[0]))
	tempOutUpstream := make(linalg.Vector, len(d.OutVecs[0]))
	tempOutUpstreamR := make(linalg.Vector, len(d.OutVecs[0]))
	for i := 0; i < len(d.OutVecs); i++ {
		if u.States != nil {
			copy(tempStateUpstream, u.States[i])
			copy(tempStateUpstreamR, u.RStates[i])
		}
		if u.Outputs != nil {
			copy(tempOutUpstream, u.Outputs[i])
			copy(tempOutUpstreamR, u.ROutputs[i])
		}
		d.OutResults[i].PropagateRGradient(tempOutUpstream, tempOutUpstreamR, rg, g)
		d.StateResults[i].PropagateRGradient(tempStateUpstream, tempStateUpstreamR, rg, g)
	}
}
