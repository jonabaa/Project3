# Functions for project 


#######################
#######################
# Heatmap plotting functions 
#######################
#######################







#######################
#######################
# Functions for reading in the data
#######################
#######################


def get_design_matrix(cleaning_function = lambda x : x, min_df = 0.0, max_df = 1.0) :
    """
    Take a data frame data, and convert to a matrix. 
    Use cleaning_function to clear up data. 
    """
    data = pd.read_json('train.json') 
    recipie_list_list = data.ingredients.values.tolist()
    recipie_string_list = [cleaning_function(" ".join(ing)) for ing in recipie_list_list]
    vectorizer = CountVectorizer(min_df = min_df, max_df = max_df)
    X = vectorizer.fit_transform(recipie_string_list)
    y = data.cuisine.values
    return X, y, vectorizer.get_feature_names()


def clean(s) :
    s = s.replace("-", " ")
    return s.replace("33", "")


#######################
#######################
# Method validation functions
#######################
#######################


def clf_cross_validator(X_train, y_train, clf_constructor, p_list, q_list = [], folds = 10, plot = False, label = '') :
    """
    A general method to preform cross validation of sci_kit learn method. 
    Takes:
    Training data (X_train, y_train)
    A function clf_constructor which builds a sci_kit learn classifier
    p_list a list of parameters to varry
    q_list a possible second list to varry, 
    folds the number of folds to use for cross validation
    plot determines if a heatmap of the results should be printed
    label is the label of the plotting
    """
    
    scores = []
    # we rename the constructor, 
    # only playes a role if q_list is empty 
    constructor = clf_constructor  
    
    ####
    # If we are only passed one list, 
    # modify the constructor to take a second dummy argument
    # We make q_list be a singleton list, 
    # and flip the role of p_list and q_list, the latter being only for printing purposes 
    if not q_list : # true if q_list is empty 
        q_list = p_list
        p_list = ['']
        constructor = (lambda p,q : clf_constructor(q)) 
    
    ###
    # We loop over the parameters 
    # and record the avarage of each folds score    
    for p in p_list :
        for q in q_list :
            clf = constructor(p,q)
            score = cross_val_score(clf, X_train, y_train, cv=folds)
            scores.append(np.mean(score))

    # transform the scores to a len(p_list) x len(q_list) shape array
    scores_array = np.array(scores).reshape(len(p_list), len(q_list))
            
    ####
    # make a heat map of the scores
    if plot :
        fig, ax = plt.subplots()
        im, cbar = heatmap(scores_array, np.array(p_list), np.array(q_list) , ax = ax, cmap = "YlGn", cbarlabel = label )
        texts = annotate_heatmap(im, valfmt="{x:.4f}")
        fig.tight_layout()
        plt.show()
        
    return scores_array

    
def svm_tester(X_train, y_train, C_list = [0.1], folds = 10, plot = False) :
    """
    Test the svm parameter C using cross validation
    For each C in C_list do folds fold cross validation, 
    returns an array, of the same size as C_list, 
    of the avarages of the accuracies for each fold.
    
    If plot is set to true, show a heatmap of the results
    """

    svm_constructor = (lambda p : svm.LinearSVC(C = p)) 
    scores = clf_cross_validator(X_train, y_train, svm_constructor, C_list, folds = folds, plot = plot, label = 'accuracy')
    return scores 

def forrest_tester(X_train, y_train, trees_list = [1], depth_list = [1], folds = 10, plot = False) :
    """
    Test the random forrest for the parameters of number of trees and max depth using cross validation
    For each pair of parameters do folds fold cross validation, 
    returns an array, of shape len(trees_list) x len(depth_list) 
    of the avarages of the accuracies for each fold.
    
    If plot is set to true, show a heatmap of the results
    """
    
    forrest_constructor = (lambda p,q : RandomForestClassifier(n_estimators = p, max_depth = q)) 
    scores = clf_cross_validator(X_train, y_train, forrest_constructor, trees_list, depth_list, folds = folds, plot = plot, label = 'accuracy')
    return scores 

def logistic_tester(X_train, y_train, C_list = [0.1], folds = 10, plot = False) :
    """
    Test the logistic regression parameter C using cross validation
    For each C in C_list do folds fold cross validation, 
    returns an array, of the same size as C_list, 
    of the avarages of the accuracies for each fold.
    
    If plot is set to true, show a heatmap of the results
    """
    
    # Not really sure about the solver
    logistic_constructor = (lambda p : LogisticRegression(solver='lbfgs', multi_class='multinomial', C = p)) 
    scores = clf_cross_validator(X_train, y_train, logistic_constructor, C_list, folds = folds, plot = plot, label = 'accuracy')
    return scores 

def mlp_tester(X_train, y_train, nodes = [1], alpha_list = [0.001], folds = 10, plot = False) :
    """
    Test the mlp classifier (neural net) for the parameters of 
    number of nodes in a single layer and regularization constant using cross validation.
    For each pair of parameters do folds fold cross validation, 
    returns an array, of shape len(nodes) x len(alpha_list) 
    of the avarages of the accuracies for each fold.
    
    If plot is set to true, show a heatmap of the results
    """
    
    mlp_constructor = (lambda p,q : MLPClassifier(hidden_layer_sizes = (p), alpha = q)) 
    scores = clf_cross_validator(X_train, y_train, mlp_constructor, nodes, alpha_list, folds = folds, plot = plot, label = 'accuracy')
    return scores


### Also want voting classifier

# Adam, implement this
def generate_design_matrix():
    design_matrix = None

    return design_matrix


# Jon, implement this
def best_accuracy_log_reg(x_train, y_train, x_test, y_test, param_list):
    best_accuracy = 0

    return best_accuracy


# Adam, implement this
def best_accuracy_svm(x_train, y_train, x_test, y_test, param_list):
    best_accuracy = 0

    return best_accuracy


# Mikael, implement this
def best_accuracy_forests(x_train, y_train, x_test, y_test, param_list):
    best_accuracy = 0

    return best_accuracy


