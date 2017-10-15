import java.util.List;

public class RadialBasisNetwork extends NeuralNetwork {

    private IActivationFunction activationFunction;
    private int numNeurons;
    private double[] means; // store a mean for each cluster

    public RadialBasisNetwork(int inputs, int outputs, int numNeurons) {
        super(inputs, outputs);
        this.activationFunction = new GaussianFunction();
        this.numNeurons = numNeurons;

        means = new double[numNeurons];
    }

    @Override
    public void train(List<Sample> samples) {
        // randomly split up data into clusters
        int index = 0;
        int sub = samples.size()/ numNeurons;
        for(int i = 0; i < numNeurons; i++){
            for(int j = 0; j < sub; j++){
                //cluster[i][j] = samples.get(index);
                index++;
            }
        }

        // calculate distances between each point in each cluster
        // reorganize data until all distances to means are minimized
        boolean swap = true;
        double[] tempMeans = new double[cluster.length];
        double maxDist = 0.0;
        while (swap == true){

            swap = false;
            for(int t = 0; t < cluster[t].length; t++) { // this is where i think i might change the data structure to arraylist.
                tempMeans[t] = calcMean(t);


                for (int k = 0; k < cluster.length; k++) {
                    for (int j = 0; j < cluster[k].length; j++) {
                        double tempDist = distance(tempMeans[0], cluster[k][j]);

                        for (int i = 1; i < tempMeans.length; i++) {
                            double temp = distance(tempMeans[i], cluster[k][j]);

                            if (temp < tempDist) {
                                //cluster[i][] = next open space;
                                swap = true;
                            }
                            if(temp > maxDist){
                                maxDist = temp;
                            }
                        }
                    }
                }
            }
        }
        //initialize network with each neuron having a mean to a cluster
        setMeans();
        for( int i = 0; i < means.length; i++){
            //new Neuron()  RBF doesn't use weights; we can initialize the neurons and just ignore the weights?
        }
        GaussianFunction.setSigma(maxDist, numNeurons);
    }

    @Override
    public double[] approximate(double[] inputs) {
        double[] outputs = new double[inputs.length];

        for( int i = 0; i < inputs.length; i++){
            double temp;
            for( int j = 0; j < means.length; j++){
                //temp = activationFunction.compute(inputs[i], means[j]);

                //if(temp )
            }
        }

        return outputs;
        // for each cluster:
        // pass the value and the mean of current cluster to the gaussian()
        // keep track of which cluster has the smallest distance between value and mean
        //

    }

    //get rid of?
    public double calcMean(int i) {
        return 0.0;
    }

    public void setMeans(){ //change to randomly select from samples
        for(int i = 0; i < numNeurons; i++){
            means[i] = calcMean(i);
        }
    }

    public double distance(double x, double y){
        return Math.pow(Math.abs(x - y), 2);
    }
}
