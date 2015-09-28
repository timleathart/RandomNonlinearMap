/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * RandomNonlinearMap.java
 */

package weka.filters.unsupervised.attribute;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.util.Random;

import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;

/**
 * <!-- globalinfo-start --> Performs a random nonlinear map on a dataset.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -N &lt;num&gt;
 *	The number of nonlinear features to generate
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Tim Leathart (tml15@students.waikato.ac.nz)
 * @version $Revision: 11516 $
 */
public class RandomNonlinearMap extends Filter implements OptionHandler,
  UnsupervisedFilter {

  /** for serialization. */
  private static final long serialVersionUID = -5649876869480249303L;

  /** The data to transform analyse/transform. */
  protected Instances m_TrainInstances;

  /** Keep a copy for the class attribute (if set). */
  protected Instances m_TrainCopy;

  /** The header for the transformed data format. */
  protected Instances m_TransformedFormat;

  /** Data has a class set. */
  protected boolean m_HasClass;

  /** Class index. */
  protected int m_ClassIndex;

  /** Number of attributes. */
  protected int m_NumAttribs;

  /** Number of instances. */
  protected int m_NumInstances;

  /** Filters for replacing missing values. */
  protected ReplaceMissingValues m_ReplaceMissingFilter;

  /** Filter for turning nominal values into numeric ones. */
  protected NominalToBinary m_NominalToBinaryFilter;

  /** Filter for removing class attribute, nominal attributes with 0 or 1 value. */
  protected Remove m_AttributeFilter;

  /** Random weight matrices **/
  protected double[][] m_weights;

  /** Random biases **/
  protected double[] m_biases;

  /** Number of nonlinear components **/
  protected int m_NumNonlinearComponents = 10;

  /** Random seed **/
  protected int m_RandomSeed = 1;

  /** The number of attributes in the pc transformed data. */
  protected int m_OutputNumAtts = -1;

  /**
   * Returns a string describing this filter.
   * 
   * @return a description of the filter suitable for displaying in the
   *         explorer/experimenter gui
   */
  public String globalInfo() {
    return "Performs a random nonlinear mapping using a randomly initialised weight matrix "
            + "followed by pointwise Cosine";

  }

  /**
   * Returns an enumeration describing the available options.
   * 
   * @return an enumeration of all the available options.
   */
  @Override
  public Enumeration<Option> listOptions() {

    Vector<Option> result = new Vector<Option>();

    result.addElement(new Option(
      "\tNumber of random nonlinear features to generate", "N", 10, "-N <num>"));

    return result.elements();
  }

  /**
   * Parses a list of options for this object.
   * <p/>
   * 
   * <!-- options-start --> Valid options are:
   * <p/>
   * 
   * <pre>
   * -N &lt;num&gt;
   *	The number of nonlinear features to generate
   * </pre>
   * 
   * <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  @Override
  public void setOptions(String[] options) throws Exception {

    String tmpStr = Utils.getOption('N', options);
    if (tmpStr.length() != 0) {
      setNumNonlinearComponents(Integer.parseInt(tmpStr));
    } else {
      setNumNonlinearComponents(10);
    }

    Utils.checkForRemainingOptions(options);
  }

  /**
   * Gets the current settings of the filter.
   * 
   * @return an array of strings suitable for passing to setOptions
   */
  @Override
  public String[] getOptions() {

    Vector<String> result = new Vector<String>();

    result.add("-N");
    result.add("" + getNumNonlinearComponents());

    return result.toArray(new String[result.size()]);
  }

  public void setNumNonlinearComponents(int value) { m_NumNonlinearComponents = value; }

  public int getNumNonlinearComponents() { return m_NumNonlinearComponents; }

  /**
   * Returns the capabilities of this evaluator.
   * 
   * @return the capabilities of this evaluator
   * @see Capabilities
   */
  @Override
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.UNARY_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.DATE_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    result.enable(Capability.NO_CLASS);

    return result;
  }

  /**
   * Determines the output format based on the input format and returns this. In
   * case the output format cannot be returned immediately, i.e.,
   * immediateOutputFormat() returns false, then this method will be called from
   * batchFinished().
   * 
   * @param inputFormat the input format to base the output format on
   * @return the output format
   * @throws Exception in case the determination goes wrong
   * @see #batchFinished()
   */
  protected Instances determineOutputFormat(Instances inputFormat)
    throws Exception {
    ArrayList<Attribute> attributes;
    int i;
    int j;

    attributes = new ArrayList<Attribute>();

    m_weights = new double[m_NumNonlinearComponents][m_NumAttribs];
    m_biases = new double[m_NumNonlinearComponents];

    Random random = new Random(m_RandomSeed);

    // Fill weights/biases matrices from uniform distribution [-1, 1]
    for(i = 0; i < m_NumNonlinearComponents; i++) {
      for (j = 0; j < m_NumAttribs; j++) {
        m_weights[i][j] = (random.nextDouble() * 2) - 1;
      }
      m_biases[i] = (random.nextDouble() * 2) - 1;
      attributes.add(new Attribute("random feature " + i));
    }

    if (m_HasClass) {
      attributes.add((Attribute) m_TrainCopy.classAttribute().copy());
    }

    Instances outputFormat = new Instances(m_TrainCopy.relationName()
            + "_random nonlinear map", attributes, 0);

    if (m_HasClass) {
      outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
    }

    m_OutputNumAtts = outputFormat.numAttributes();

    return outputFormat;
  }

  /**
   * Transform an instance in original (unormalized) format.
   * 
   * @param instance an instance in the original (unormalized) format
   * @return a transformed instance
   * @throws Exception if instance can't be transformed
   */
  protected Instance convertInstance(Instance instance) throws Exception {
    Instance result;
    double[] newVals;
    int i;
    int j;

    System.out.println(m_HasClass);

    newVals = new double[m_OutputNumAtts];

    for (j = 0; j < m_NumNonlinearComponents; j++) {
      double sum = m_biases[j];

      for (i = 0; i < m_NumAttribs; i++) {
        sum += m_weights[j][i] * instance.value(i);
      }

      newVals[j] = Math.cos(sum);
    }

    if (m_HasClass) {
      newVals[j] = instance.value(instance.classIndex());
    }

    // create instance
    if (instance instanceof SparseInstance) {
      result = new SparseInstance(instance.weight(), newVals);
    } else {
      result = new DenseInstance(instance.weight(), newVals);
    }

    return result;
  }

  /**
   * Initializes the filter with the given input data.
   * 
   * @param instances the data to process
   * @throws Exception in case the processing goes wrong
   * @see #batchFinished()
   */
  protected void setup(Instances instances) throws Exception {
    int i;
    Vector<Integer> deleteCols;
    int[] todelete;

    m_TrainInstances = new Instances(instances);

    // make a copy of the training data so that we can get the class
    // column to append to the transformed data (if necessary)
    m_TrainCopy = new Instances(m_TrainInstances, 0);

    m_ReplaceMissingFilter = new ReplaceMissingValues();
    m_ReplaceMissingFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances,
      m_ReplaceMissingFilter);

    m_NominalToBinaryFilter = new NominalToBinary();
    m_NominalToBinaryFilter.setInputFormat(m_TrainInstances);
    m_TrainInstances = Filter.useFilter(m_TrainInstances,
      m_NominalToBinaryFilter);

    // delete any attributes with only one distinct value or are all missing
    deleteCols = new Vector<Integer>();
    for (i = 0; i < m_TrainInstances.numAttributes(); i++) {
      if (m_TrainInstances.numDistinctValues(i) <= 1) {
        deleteCols.addElement(i);
      }
    }

    if (m_TrainInstances.classIndex() >= 0) {
      // get rid of the class column
      m_HasClass = true;
      m_ClassIndex = m_TrainInstances.classIndex();
      deleteCols.addElement(new Integer(m_ClassIndex));
    }

    // remove columns from the data if necessary
    if (deleteCols.size() > 0) {
      m_AttributeFilter = new Remove();
      todelete = new int[deleteCols.size()];
      for (i = 0; i < deleteCols.size(); i++) {
        todelete[i] = (deleteCols.elementAt(i)).intValue();
      }
      m_AttributeFilter.setAttributeIndicesArray(todelete);
      m_AttributeFilter.setInvertSelection(false);
      m_AttributeFilter.setInputFormat(m_TrainInstances);
      m_TrainInstances = Filter.useFilter(m_TrainInstances, m_AttributeFilter);
    }

    // can evaluator handle the processed data ? e.g., enough attributes?
    getCapabilities().testWithFail(m_TrainInstances);

    m_NumInstances = m_TrainInstances.numInstances();
    m_NumAttribs = m_TrainInstances.numAttributes();

    m_TransformedFormat = determineOutputFormat(m_TrainInstances);
    setOutputFormat(m_TransformedFormat);

    m_TrainInstances = null;
  }

  /**
   * Sets the format of the input instances.
   * 
   * @param instanceInfo an Instances object containing the input instance
   *          structure (any instances contained in the object are ignored -
   *          only the structure is required).
   * @return true if the outputFormat may be collected immediately
   * @throws Exception if the input format can't be set successfully
   */
  @Override
  public boolean setInputFormat(Instances instanceInfo) throws Exception {
    super.setInputFormat(instanceInfo);

    m_OutputNumAtts = -1;
    m_AttributeFilter = null;
    m_NominalToBinaryFilter = null;

    return false;
  }

  /**
   * Input an instance for filtering. Filter requires all training instances be
   * read before producing output.
   * 
   * @param instance the input instance
   * @return true if the filtered instance may now be collected with output().
   * @throws IllegalStateException if no input format has been set
   * @throws Exception if conversion fails
   */
  @Override
  public boolean input(Instance instance) throws Exception {
    Instance inst;

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }

    if (isNewBatch()) {
      resetQueue();
      m_NewBatch = false;
    }

    if (isFirstBatchDone()) {
      inst = convertInstance(instance);
      inst.setDataset(getOutputFormat());
      push(inst);
      return true;
    } else {
      bufferInput(instance);
      return false;
    }
  }

  /**
   * Signify that this batch of input to the filter is finished.
   * 
   * @return true if there are instances pending output
   * @throws NullPointerException if no input structure has been defined,
   * @throws Exception if there was a problem finishing the batch.
   */
  @Override
  public boolean batchFinished() throws Exception {
    int i;
    Instances insts;
    Instance inst;

    if (getInputFormat() == null) {
      throw new NullPointerException("No input instance format defined");
    }

    insts = getInputFormat();

    if (!isFirstBatchDone()) {
      setup(insts);
    }

    for (i = 0; i < insts.numInstances(); i++) {
      inst = convertInstance(insts.instance(i));
      inst.setDataset(getOutputFormat());
      push(inst);
    }

    flushInput();
    m_NewBatch = true;
    m_FirstBatchDone = true;

    return (numPendingOutput() != 0);
  }

  /**
   * Returns the revision string.
   * 
   * @return the revision
   */
  @Override
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 11516 $");
  }

  /**
   * Main method for running this filter.
   * 
   * @param args should contain arguments to the filter: use -h for help
   */
  public static void main(String[] args) {
    runFilter(new RandomNonlinearMap(), args);
  }
}
