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
	// change between timesteps.
	// Either this will contain one value (a single rate)
	// or multiple values (one rate per weight).
	// The rate is squashed between 0 and 1 before being
	// used, where 0 means no change and 1 means complete
	// change.
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
// If variableRate is true, a different trace rate is used
// for each weight in the layer.
func NewDenseLayer(inCount, outCount int, variableRate bool) *DenseLayer {
	weightCount := inCount * outCount
	traceCount := 1
	if variableRate {
		traceCount = weightCount
	}
	res := &DenseLayer{
		InputCount:   inCount,
		OutputCount:  outCount,
		TraceRate:    &autofunc.Variable{Vector: make(linalg.Vector, traceCount)},
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

// InitRates randomly initializes the trace rates in a
// biased fashion.
// The rates are divided up into three sections: long-term
// short-term, and neutral.
// The arguments specify, out of all rates, the fraction
// of long-term and short-term ones.
func (d *DenseLayer) InitRates(longTerm, shortTerm float64) {
	indices := rand.Perm(len(d.TraceRate.Vector))
	lt := int(math.Ceil(longTerm * float64(len(indices))))
	st := int(math.Ceil(shortTerm * float64(len(indices))))
	for lt+st > len(indices) {
		if st > 0 {
			st--
		} else {
			lt--
		}
	}
	for _, i := range indices[:lt] {
		d.TraceRate.Vector[i] = rand.Float64() - 2
	}
	for _, i := range indices[lt:st] {
		d.TraceRate.Vector[i] = rand.Float64() + 2
	}
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
func (d *DenseLayer) StartState() rnn.State {
	return rnn.VecState(d.InitTrace.Vector)
}

// StartRState returns the initial trace.
func (d *DenseLayer) StartRState(rv autofunc.RVector) rnn.RState {
	rvar := autofunc.NewRVariable(d.InitTrace, rv)
	return rnn.VecRState{State: rvar.Output(), RState: rvar.ROutput()}
}

// PropagateStart propagates through the start state.
func (d *DenseLayer) PropagateStart(_ []rnn.State, s []rnn.StateGrad, g autofunc.Gradient) {
	rnn.PropagateVarState(d.InitTrace, s, g)
}

// PropagateStartR propagates through the start state.
func (d *DenseLayer) PropagateStartR(_ []rnn.RState, s []rnn.RStateGrad, rg autofunc.RGradient,
	g autofunc.Gradient) {
	rnn.PropagateVarStateR(d.InitTrace, s, rg, g)
}

// ApplyBlock applies the layer to a batch of inputs.
func (d *DenseLayer) ApplyBlock(s []rnn.State, in []autofunc.Result) rnn.BlockResult {
	res := &denseLayerOutput{}
	res.StatePool, _ = rnn.PoolVecStates(s)
	for i, input := range in {
		newState, out := d.timestep(res.StatePool[i], input)
		res.StateResults = append(res.StateResults, newState)
		res.OutResults = append(res.OutResults, out)
		res.StatesOut = append(res.StatesOut, rnn.VecState(newState.Output()))
		res.VecsOut = append(res.VecsOut, out.Output())
	}
	return res
}

// BatchR applies the layer to a set of inputs.
func (d *DenseLayer) ApplyBlockR(rv autofunc.RVector, s []rnn.RState,
	in []autofunc.RResult) rnn.BlockRResult {
	res := &denseLayerROutput{}
	var pool []autofunc.RResult
	res.StatePool, pool = rnn.PoolVecRStates(s)
	for i, input := range in {
		newState, out := d.timestepR(rv, pool[i], input)
		res.StateResults = append(res.StateResults, newState)
		res.OutResults = append(res.OutResults, out)
		res.VecsOut = append(res.VecsOut, out.Output())
		res.RVecsOut = append(res.RVecsOut, out.ROutput())
		res.StatesOut = append(res.StatesOut, rnn.VecRState{
			State:  newState.Output(),
			RState: newState.ROutput(),
		})
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

	traceRate := neuralnet.Sigmoid{}.Apply(d.TraceRate)
	keepRate := autofunc.AddScaler(autofunc.Scale(traceRate, -1), 1)
	if len(keepRate.Output()) == 1 {
		newState = autofunc.Add(autofunc.ScaleFirst(state, keepRate),
			autofunc.ScaleFirst(autofunc.OuterProduct(out, in), traceRate))
	} else {
		newState = autofunc.Add(autofunc.Mul(state, keepRate),
			autofunc.Mul(autofunc.OuterProduct(out, in), traceRate))
	}

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

	traceRate := neuralnet.Sigmoid{}.ApplyR(rv, autofunc.NewRVariable(d.TraceRate, rv))
	keepRate := autofunc.AddScalerR(autofunc.ScaleR(traceRate, -1), 1)
	if len(keepRate.Output()) == 1 {
		newState = autofunc.AddR(autofunc.ScaleFirstR(state, keepRate),
			autofunc.ScaleFirstR(autofunc.OuterProductR(out, in), traceRate))
	} else {
		newState = autofunc.AddR(autofunc.MulR(state, keepRate),
			autofunc.MulR(autofunc.OuterProductR(out, in), traceRate))
	}

	return
}

type denseLayerOutput struct {
	StatePool    []*autofunc.Variable
	VecsOut      []linalg.Vector
	StatesOut    []rnn.State
	OutResults   []autofunc.Result
	StateResults []autofunc.Result
}

func (d *denseLayerOutput) Outputs() []linalg.Vector {
	return d.VecsOut
}

func (d *denseLayerOutput) States() []rnn.State {
	return d.StatesOut
}

func (d *denseLayerOutput) PropagateGradient(u []linalg.Vector, s []rnn.StateGrad,
	g autofunc.Gradient) []rnn.StateGrad {
	if len(d.StatesOut) == 0 {
		return nil
	}
	return rnn.PropagateVecStatePool(g, d.StatePool, func() {
		tempStateUpstream := make(linalg.Vector, len(d.StatesOut[0].(rnn.VecState)))
		tempOutUpstream := make(linalg.Vector, len(d.VecsOut[0]))
		for i := range d.VecsOut {
			if s != nil && s[i] != nil {
				copy(tempStateUpstream, s[i].(rnn.VecStateGrad))
			} else if s != nil && i > 0 && s[i-1] != nil {
				for j := range tempStateUpstream {
					tempStateUpstream[j] = 0
				}
			}
			if u != nil {
				copy(tempOutUpstream, u[i])
			}
			d.OutResults[i].PropagateGradient(tempOutUpstream, g)
			d.StateResults[i].PropagateGradient(tempStateUpstream, g)
		}
	})
}

type denseLayerROutput struct {
	StatePool    []*autofunc.Variable
	VecsOut      []linalg.Vector
	RVecsOut     []linalg.Vector
	StatesOut    []rnn.RState
	OutResults   []autofunc.RResult
	StateResults []autofunc.RResult
}

func (d *denseLayerROutput) Outputs() []linalg.Vector {
	return d.VecsOut
}

func (d *denseLayerROutput) ROutputs() []linalg.Vector {
	return d.RVecsOut
}

func (d *denseLayerROutput) RStates() []rnn.RState {
	return d.StatesOut
}

func (d *denseLayerROutput) PropagateRGradient(u, uR []linalg.Vector, s []rnn.RStateGrad,
	rg autofunc.RGradient, g autofunc.Gradient) []rnn.RStateGrad {
	if len(d.StatesOut) == 0 {
		return nil
	}
	return rnn.PropagateVecRStatePool(rg, g, d.StatePool, func() {
		tempStateUpstream := make(linalg.Vector, len(d.StatesOut[0].(rnn.VecRState).State))
		tempStateUpstreamR := make(linalg.Vector, len(tempStateUpstream))
		tempOutUpstream := make(linalg.Vector, len(d.VecsOut[0]))
		tempOutUpstreamR := make(linalg.Vector, len(d.VecsOut[0]))
		for i := 0; i < len(d.VecsOut); i++ {
			if s != nil && s[i] != nil {
				copy(tempStateUpstream, s[i].(rnn.VecRStateGrad).State)
				copy(tempStateUpstreamR, s[i].(rnn.VecRStateGrad).RState)
			} else if s != nil && i > 0 && s[i-1] != nil {
				for j := range tempStateUpstream {
					tempStateUpstream[j] = 0
					tempStateUpstreamR[j] = 0
				}
			}
			if u != nil {
				copy(tempOutUpstream, u[i])
				copy(tempOutUpstreamR, uR[i])
			}
			d.OutResults[i].PropagateRGradient(tempOutUpstream, tempOutUpstreamR, rg, g)
			d.StateResults[i].PropagateRGradient(tempStateUpstream, tempStateUpstreamR, rg, g)
		}
	})
}
