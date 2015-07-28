//package ml_decision_trees;

/*
 * Author: Vivek Anand Sampath
 * NetID: vxs135130
 */


import java.io.*;
import java.util.*;

public class decision_tree {
	public static int[][] trn_data_set;
	public static int[][] val_data_set;
	public static int[][] tst_data_set;
	public static String[] attributes;
	public static int heuristic = 1;
	
	/*
	 * INPUT: file_name
	 * OUTPUT: number of lines the file
	 * DESCRIPTION: find the number of lines in the file passed discarding the header row
	 */
	public static int getLinesInFile(String file_name){
		int noLines = 0;
		FileReader fr;
		String ln = new String();
		try{
			fr = new FileReader(file_name);
			BufferedReader br = new BufferedReader(fr);
			while((ln = br.readLine()) != null){
				noLines++;
				if(noLines == 1){
					attributes = ln.split(",");
				}
			}
			br.close();
			fr.close();
		} catch(FileNotFoundException e1){
			System.out.println(e1.getMessage());
		} catch(Exception e2){
			System.out.println("Error in opening file "+file_name);
		}
		
		return (noLines-1);
	}
	
	/*
	 * INPUT: file_name
	 * OUTPUT: data_set (2D array of the data set)
	 * DESCRIPTION: reads the file with passed in name(.csv file) and stores the data set into 2D array
	 */
	public static int[][] readDataSet(String file_name){
		int noLines;
		String ln = new String();
		String attribute_values[] = null;
		FileReader fr;
		int lineNumber = 0, row = 0;
		int[][] data_set;
		
		int numLines = getLinesInFile(file_name);
		data_set = new int[numLines][attributes.length];
		
		row = 0;
		try{
			fr = new FileReader(file_name);
			BufferedReader br = new BufferedReader(fr);
			while((ln = br.readLine()) != null){
				row++;
				if(row>1){
					attribute_values = ln.split(",");
					for(int i=0;i<attribute_values.length;i++){
						data_set[row-2][i] = Integer.parseInt(attribute_values[i]);
					}
				}
			}
		} catch(FileNotFoundException e1){
			System.out.println("Error in opening file "+file_name);
		} catch(IOException e2){
			System.out.println("IO Exception");
		}
		
		return data_set;
		
	}
	
	/*
	 * INPUT: data_set
	 * OUTPUT: number of positive examples in the data set
	 * DESCRIPTION: Check for each of the value of the ClassName attribute in the data_set and return the number of +ve examples
	 */
	public static int getPositiveCount(int[][] data_set){
		int numPos = 0;
		int targetAttribute = attributes.length - 1;
		for(int i=0;i<data_set.length;i++){
			if(data_set[i][targetAttribute] == 1){
				numPos++;
			}
		}
		return numPos;
	}
	
	/*
	 * INPUT: 	num_pos - number of positive examples 
	 * 			num_neg - number of negative examples
	 * OUTPUT: entropy of the example set for given distribution/proportion
	 */
	public static double entropy(int num_pos, int num_neg){
		double pos_proportion, neg_proportion; 
		double ent = 0;
		if((num_pos+num_neg)>0){
			pos_proportion = (double)(num_pos)/(num_pos + num_neg);
			neg_proportion = (double)(1 - pos_proportion);
			
			if(pos_proportion == 0){
				ent = (neg_proportion * Math.log(neg_proportion)) * (-1);
			} else if(neg_proportion == 0){
				ent = (pos_proportion * Math.log(pos_proportion)) * (-1);
			} else {
				ent = (pos_proportion * Math.log(pos_proportion) + neg_proportion * Math.log(neg_proportion)) * (-1);
			}
		}
		
		return ent;
	}
	
	/*
	 * INPUT:	num_pos - number of positive examples
	 * 			num_neg - number of negative examples
	 * OUTPUT: finds the variance impurity for the given distribution of set
	 */
	public static double variance_impurity(int num_pos, int num_neg){
		double pos_proportion, neg_proportion;
		double im = 0;
		double total = num_pos+num_neg;
		
		if(total > 0){
			im = (num_pos * num_neg * 1.0)/(total * total);
		}
		
		return im;
	}
	
	/*
	 * INPUT: 	data_set - the data set of examples for which the best attribute for classification is to be found,
	 * 			chose_attributes - list of attributes to chose from
	 * OUTPUT: best_attribute
	 * DESCRIPTION: find the information gain for each attribute for the given data set and return the one attribute with
	 * maximum gain 
	 */
	public static String getBestClassifier(int[][] data_set, String[] chose_attributes){
		
		int[] attribute_values = new int[]{0,1};
		int[][] subset_data = null;
		int subset_pos0, subset_neg0, subset_total0;
		int subset_pos1, subset_neg1, subset_total1;
		double[] subset_ent = new double[2];
		String best_attribute = null;
		double gain, max_gain;
		double ent1 = 0, ent2 = 0;
		double var1 = 0, var2 = 0;
		
		int total = data_set.length;
		int num_pos = getPositiveCount(data_set);
		int num_neg = total - num_pos;
		double ent = entropy(num_pos, num_neg);
		
		max_gain = -1;
		if(chose_attributes.length > 1){
			for(int i=0;i<chose_attributes.length;i++){
				subset_data = getExamplesSubset(data_set, chose_attributes[i], 0);
				
				subset_total0 = subset_data.length;
				subset_pos0 = getPositiveCount(subset_data);
				subset_neg0 = subset_total0 - subset_pos0;
				
				subset_data = getExamplesSubset(data_set, chose_attributes[i], 1);
				
				subset_total1 = subset_data.length;
				subset_pos1 = getPositiveCount(subset_data);
				subset_neg1 = subset_total1 - subset_pos1;
				
				if(heuristic == 0){
					ent1 = (double)entropy(subset_pos0, subset_neg0);
					ent2 = (double)entropy(subset_pos1, subset_neg1);
					
					gain = ent - (((subset_total0*1.0)/total) * ent1) - (((subset_total1*1.0)/total) * ent2);
				} else {
					var1 = (double)variance_impurity(subset_pos0, subset_neg0);
					var2 = (double)variance_impurity(subset_pos1, subset_neg1);
					
					gain = ent - (((subset_total0*1.0)/total) * var1) - (((subset_total1*1.0)/total) * var2);
				}
				
				if(gain > max_gain){
					max_gain = gain;
					best_attribute = new String(chose_attributes[i]);
				}
			}
		} else if(chose_attributes.length == 1){
			best_attribute = chose_attributes[0];
		}
		
		return best_attribute;
	}
	
	/*
	 * INPUT: 	data_set,
	 * 			attr_name - attribute for which the value is substituted
	 * 			attr_value - attribute value to assign for the attribute passed
	 * OUTPUT: subset_data_set - data set passed in is filtered for the given attr_name = attr_value combination and the resulting set
	 * is returned
	 */
	public static int[][] getExamplesSubset(int[][] data_set, String attr_name, int attr_value){
		int attributeIndex = 0;
		int[][] subset_data;
		int count=0;
		
		for(int i=0;i<attributes.length;i++){
			if(attributes[i].equals(attr_name)){
				attributeIndex = i;
				break;
			}
		}
		
		int noExamples = 0;
		for(int i=0;i<data_set.length;i++){
			if(data_set[i][attributeIndex] == attr_value){
				noExamples++;
			}
		}
		
		subset_data = new int[noExamples][attributes.length-1];
		for(int i=0;i<data_set.length;i++){
			if(data_set[i][attributeIndex] == attr_value){
				subset_data[count] = data_set[i];
				count++;
			}
		}
		
		return subset_data;
	}
	
	/*
	 * INPUT: 	data_set,
	 * 			target_attribute - attribute we are trying to predict,
	 * 			input_attributes - features used for predicting the target_attribute
	 * OUTPUT: DecisionTree - header node for the decision tree for the given data set
	 */
	public static DecisionTree ID3_algorithm(int[][] data_set, String target_attribute, String[] input_attributes){
		DecisionTree d = new DecisionTree();
		
		d.total = data_set.length;
		d.num_pos = getPositiveCount(data_set);
		d.num_neg = d.total - d.num_pos;
		
		d.node_attribute = null;
		d.dl = null;
		d.dr = null;
		
		if(d.num_pos == d.total){
			d.node_value = 1;
			return d;
		} else if (d.num_neg == d.total){
			d.node_value = 0;
			return d;
		}
		
		if(input_attributes.length == 0){
			if(d.num_pos >= d.num_neg){
				d.node_value = 1;
				return d;
			} else {
				d.node_value = 0;
				return d;
			}
		}
		
		String best_attribute = getBestClassifier(data_set, input_attributes);
		d.node_attribute = best_attribute;
		String[] new_attributes = new String[input_attributes.length - 1];
		int count = 0;
		for(int j=0;j<input_attributes.length;j++){
			if(!input_attributes[j].equals(best_attribute)){
				new_attributes[count] = input_attributes[j];
				count++;
			}
		}
		int[] attribute_values = new int[]{0,1};
		for(int i=0;i<attribute_values.length;i++){
			int[][] sub_data_set = getExamplesSubset(data_set, best_attribute, attribute_values[i]);
			if(data_set.length == 0){
				DecisionTree child = new DecisionTree();
				child.num_pos = 0;
				child.num_neg = 0;
				child.total = 0;
				child.node_attribute = null;
				
				if(d.num_pos >= d.num_neg){
					child.node_value = 1;
					d.dr = child;
				} else {
					child.node_value = 0;
					d.dl = child;
				}
			} else {
				DecisionTree child = ID3_algorithm(sub_data_set, target_attribute, new_attributes);
				if(attribute_values[i] == 0){
					d.dl = child;
				} else {
					d.dr = child;
				}
			}
		}
		
		return d;
	}

	/*
	 * INPUT: 	DecisionTree - header node for the decision tree,
	 * 			depth - depth of the recursive call
	 * OUTPUT: print the decision tree structure
	 */
	public static void traverseDecisionTree(DecisionTree node, int depth){
		if(node.dl == null && node.dr == null){
			System.out.print(node.node_value);
			System.out.println();
		} else {
			System.out.println();
			if(node.dl != null){
				for(int i=0;i<depth;i++){
					System.out.print("|");
				}
				System.out.print(node.node_attribute);
				System.out.print(" = 0 :");
				traverseDecisionTree(node.dl, depth+1);
			}
			
			if(node.dr != null){
				for(int i=0;i<depth;i++){
					System.out.print("|");
				}
				System.out.print(node.node_attribute);
				System.out.print(" = 1 :");
				traverseDecisionTree(node.dr, depth+1);
			}
		}
	}
	
	/*
	 * INPUT: DecisionTree - root node for the decision tree
	 * OUTPUT: copy of the decision tree passed
	 */
	public static DecisionTree copyDecisionTree(DecisionTree node){
		if(node.dl == null && node.dr == null){
			DecisionTree n = new DecisionTree();
			n.node_attribute = node.node_attribute;
			n.node_value = node.node_value;
			n.total = node.total;
			n.num_pos = node.num_pos;
			n.num_neg = node.num_neg;
			n.dl = null;
			n.dr = null;
			return n;
		} else {
			DecisionTree n = new DecisionTree();
			n.node_attribute = node.node_attribute;
			n.node_value = node.node_value;
			n.total = node.total;
			n.num_pos = node.num_pos;
			n.num_neg = node.num_neg;
			n.dl = null;
			n.dr = null;
			if(node.dl != null){
				n.dl = copyDecisionTree(node.dl);
			}
			
			if(node.dr != null){
				n.dr = copyDecisionTree(node.dr);
			}
			
			return n; 
		}
	}
	
	/*
	 * INPUT: DecisionTree - root node for the decision tree
	 * OUTPUT: number the non-leaf nodes of the decision tree
	 */
	public static int numberDecisionTree(DecisionTree root, int num){
		int lnum = 0, rnum = 0, nextNum;
		if(root.dl == null && root.dr == null){
			root.order = 0;
			return 0;
		} else {
			if(root.dl != null){
				lnum = numberDecisionTree(root.dl, num);
			}
			
			if(lnum == 0){
				nextNum = num;
			} else {
				nextNum = lnum; 
			}
			root.order = nextNum++;
			
			if(root.dr != null){
				rnum = numberDecisionTree(root.dr, nextNum);
				if(rnum != 0){
					nextNum = rnum;
				}
			}
			
			return nextNum;
		}
	}
	
	/*
	 * Replace the node passed in with a non-leaf node with value equal to the most common value of the data set 
	 * associated with that node
	 */
	public static void replaceNodeAt(DecisionTree d, int node_to_replace){
		while(true){
			if(node_to_replace < d.order && d.dl != null){
				d = d.dl;
			} else if(node_to_replace > d.order && d.dr != null) {
				d = d.dr;
			} else if(node_to_replace == d.order || (d.dl == null && d.dr == null)) {
				break;
			} else {
				break;
			}
		}
		
		if(d.num_pos > d.num_neg){
			d.node_value = 1;
			d.node_attribute = null;
			d.dl = null;
			d.dr = null;
		} else if(d.num_pos < d.num_neg){
			d.node_value = 0;
			d.node_attribute = null;
			d.dl = null;
			d.dr = null;
		} else {
//			System.out.println("same proportion: couldn't decide");
		}
	}
	
	/*
	 * INPUT: 	DecisionTree - root node of the decision tree,
	 * 			data_set_identifier - data set(training, validation (or) test) 
	 * OUTPUT: accuracy of the decision tree for the data set passed in percentage
	 */
	public static double calculateAccuracy(DecisionTree node, String data_set_identifier){
		String attr_name;
		int attributeIndex = 0;
		
		int[][] set;
		if(data_set_identifier.equals("validation")){
			set = val_data_set;
		} else if(data_set_identifier.equals("test")){
			set = tst_data_set;
		} else{
			set = trn_data_set;
		}
		
		int cnt=0;
		
		for(int i=0;i<set.length;i++){
			DecisionTree traverse_node = node;
			while(traverse_node.dl != null && traverse_node.dr != null){
				attr_name = new String(traverse_node.node_attribute);
				for(int j=0;j<attributes.length;j++){
					if(attributes[j].equals(attr_name)){
						attributeIndex = j;
						break;
					}
				}
				
				if(set[i][attributeIndex] == 0){
					traverse_node = traverse_node.dl;
				} else {
					traverse_node = traverse_node.dr;
				}
			}
			
			if(traverse_node.node_value == set[i][attributes.length-1]){
				cnt++;
			}
		}
		
		return (double)((cnt*1.0)/set.length)*100.0;
	}
	
	/*
	 * INPUT: 	DecisionTree - root node,
	 * 			l - number of iterations for pruning
	 * 			k - maximum number of nodes to prune
	 * OUTPUT: DecisionTree - pruned
	 */
	public static DecisionTree post_pruning(DecisionTree d, int l, int k){
		int m, n, max_node;
		Random rand = new Random();
		double accuracy, accuracy_1;
		
		DecisionTree best_d = copyDecisionTree(d);
		accuracy = calculateAccuracy(best_d, "validation");
		max_node = (numberDecisionTree(best_d, 1) - 1);
		for(int i=0;i<l;i++){
			System.gc();
			DecisionTree d_1 = copyDecisionTree(d);
			while(true){
				m = rand.nextInt(k+1);
				if(m != 0){
					break;
				}
			}
			for(int j=0;j<m;j++){
				max_node = (numberDecisionTree(d_1, 1) - 1);
				if(max_node>0){
					while(true){
						n = rand.nextInt(max_node+1);
						if(n != 0){
							break;
						}
					}
					replaceNodeAt(d_1, n);
				} else {
					break;
				}
			}

			accuracy_1 = calculateAccuracy(d_1, "validation");
			if(accuracy_1 > accuracy){
				best_d = copyDecisionTree(d_1);
				accuracy = accuracy_1;
			}
		}
		
		return best_d;
	}
	
	/*
	 * INPUT: Command line arguments
	 * l - post-pruning parameter
	 * k - post-pruning parameter
	 * trn_set_file - training data set file location
	 * val_set_file - validation data set file location
	 * tst_set_file - test data set file location
	 * print_tree - yes|no parameter to print the decision tree
	 * 
	 *  OUTPUT: Construct decision tree, do post-pruning based on l,k parameters.
	 *  Find the accuracy of the decision tree on test data set.
	 *  Print the decision tree based on print_tree parameter.
	 */
	public static void main(String[] args){
		int l;
		int k;
		String trn_set_file;
		String val_set_file;
		String tst_set_file;
		String print_tree;
		double accuracy;
		
		if(args.length == 6){
			l = Integer.parseInt(args[0]);
			k = Integer.parseInt(args[1]);
			trn_set_file = new String(args[2]);
			val_set_file = new String(args[3]);
			tst_set_file = new String(args[4]);
			print_tree = new String(args[5]);
		} else {
			System.out.println("Invalid command line arguments passed. Program halted.");
			return;
			/*
			l = 10;
			k = 5;
			trn_set_file = new String("/home/vivek/Google Drive/utd/07 machine learning/Vibhav projects/hw1/ds1/training_set.csv");
			val_set_file = new String("/home/vivek/Google Drive/utd/07 machine learning/Vibhav projects/hw1/ds1/validation_set.csv");
			tst_set_file = new String("/home/vivek/Google Drive/utd/07 machine learning/Vibhav projects/hw1/ds1/test_set.csv");
			print_tree = new String("no");
			*/
		}
		
		// set global attributes and returns attribute values read from the file
		trn_data_set = readDataSet(trn_set_file);
		val_data_set = readDataSet(val_set_file);
		tst_data_set = readDataSet(tst_set_file);
		
		if(trn_data_set.length == 0){
			// there are no examples
			return;
		}
		
		DecisionTree d;
		
		
		String[] heuristic_string = new String[]{"Entropy","Variance Impurity"};
		for(int h=0;h<2;h++){
			heuristic = h;
			
			System.out.println();
			System.out.println("HEURISTIC : "+heuristic_string[h]);
			System.out.println();
			
			String[] copy_attributes = new String[attributes.length-1];
			for(int i=0;i<copy_attributes.length;i++){
				copy_attributes[i] = attributes[i];
			}
			
			d = ID3_algorithm(trn_data_set, attributes[attributes.length-1], copy_attributes);
			
			Double accuracy_bef = calculateAccuracy(d, "test");
			
			DecisionTree best_d = post_pruning(d, l, k);
			
			if(print_tree.equals("yes")){
				System.out.println();
				System.out.println("DECISION TREE (after Post-Pruning):");
				traverseDecisionTree(best_d,0);
			}
			accuracy = calculateAccuracy(best_d, "test");
			System.out.println("Accuracy - before Post-Pruning : "+accuracy_bef);
			System.out.println("Accuracy - after Post-Pruning(L="+l+", K="+k+") : "+accuracy);
		}
	}
}

class DecisionTree{
	int total;
	int num_pos;
	int num_neg;
	int node_value;
	int order;
	String node_attribute;
	DecisionTree dl,dr;
}
