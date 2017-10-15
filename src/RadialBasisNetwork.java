import java.util.List;
import java.util.Random;

public class RadialBasisNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private int numNeurons;
    private Sample[] means; // store a mean for each cluster
    private double learnRate;
    private double[] errors; // average errors from each gradient descent iteration
    private int batchSize;
    private int epochs;
    private Layer layer;

    public RadialBasisNetwork(int inputs, int outputs, int numNeurons, double learnRate, int batchsize, int epochs) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.numNeurons = numNeurons;
        this.learnRate = learnRate;
        this.batchSize = batchsize;
        this.epochs = epochs;
        means = new Sample[numNeurons];

    }

    @Override
    public void train(List<Sample> samples) {
        setMeans(samples);

        layer = new Layer(numNeurons, 1, activationFunction);

        layer = new Layer(numNeurons, 1, null);

        for (int i = 0; i < layer.size; i++){
            layer.getNeuron(i).setOutput(means[i].inputs[0]); //using outputs to store the input to compare with the
        }
        System.out.println(layer.getNeuron(0).getOutput() + " " + layer.getNeuron(1).getOutput());

        double maxDist = distance(layer.getNeuron(0).getOutput(), layer.getNeuron(1).getOutput());

        for(int i = 0; i < layer.size; i ++){
            for(int j = i + 1; j < layer.size; j++){
                double dist = distance(layer.getNeuron(i).getOutput(), layer.getNeuron(j).getOutput());
                if(dist > maxDist){
                    maxDist = dist;
                }
            }
        }

        GaussianFunction.setSigma(maxDist, numNeurons);

        for (int i = 0; i < epochs; i++) {
            double epochError = 0.0;
            double[] networkOutputs = new double[samples.size()];
            int k = 0;
            for (Sample sample : samples) {
                double[] gaussOutputs = gaussian(sample.inputs);
                networkOutputs[k] = weightedSum(gaussOutputs);
                if(k % batchSize == 0) {
                    updateWeights(k, sample.inputs);
                }
                epochError += this.calculateTotalError(sample.outputs, networkOutputs);
                k++;
            }
            System.out.println("Epoch: " + i + "\t\tError: " + epochError / samples.size());
        }
    }

    @Override
    public double[] approximate(double[] inputs) {
        double[] outputs = new double[inputs.length];
        outputs = gaussian(inputs);

        double[] approx = new double[inputs.length];

        for(int i = 0; i < inputs.length; i++){
            approx[i] = weightedSum(outputs);
        }

        return approx;
        // pass input thru the gaussians
        //weighted sum as output


    }


    public void setMeans(List<Sample> sample){ //change to randomly select from samples

        for(int i = 0; i < numNeurons; i++){
            Random random = new Random();
            int temp = (int) (random.nextDouble()* sample.size() / (i + 1));
            //System.out.println("temp index: " + temp);
            means[i] = sample.get((int) temp );

        }
    }

    public double distance(double x, double y){
        return Math.pow(Math.abs(x - y), 2);
    }

    public void updateWeights(int index, double[] inputs){
        //update the weights of each neuron

        for(int i = index; i < i + batchSize && i < layer.size; i++){
            layer.getNeuron(i).updateWeight(0, learnRate * activationFunction.computeDerivative(distance(inputs[0], means[i].inputs[0])));
        }
    }

    public double calculateTotalError(double[] networkOutputs, double[] expectedOutputs){
        assert networkOutputs.length == expectedOutputs.length;

        double errorSum = 0.0;
        // Calculate the sum over the squared error for each output value
        for (int i = 0; i < networkOutputs.length; i++) {
            double error = networkOutputs[i] - expectedOutputs[i];
            errorSum += Math.pow(error, 2);
        }

        // Normalize and return error
        return errorSum / (networkOutputs.length * expectedOutputs.length);
    }

    public double[] gaussian(double[] inputs){

        double[] gauss = new double[layer.size];
        for( int i = 0; i < layer.size; i++){
            gauss[i] = layer.getNeuron(i).getWeight(0) * activationFunction.compute(distance(inputs[0], means[i].inputs[0]));
        }
        return gauss;
    }

    public double weightedSum(double[] gaussOutput){
        double weighted = 0.0;
        for(int i = 0; i < gaussOutput.length; i++){
            weighted += gaussOutput[i];
        }
        return weighted;
    }
}
