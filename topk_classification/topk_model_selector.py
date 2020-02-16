
def topk_performance(model,n_topk,x_train,y_train,x_test,y_test,yencoder):
    """
    
    """
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    cat_class = yencoder.inverse_transform(model.classes_)
    prob_vals = model.predict_proba(x_test)
    sort_ind = argsort(prob_vals, axis=1)[:,-n_topk:]
    prob_sort = array([x[i] for i,x in zip(sort_ind,prob_vals)])
    m_class_sort = model.classes_[sort_ind]
    class_sort = cat_class[sort_ind]
    accuracy = [0. for i in range(n_topk+1)]
    for_tup = [0. for i in range(n_topk+1)]
    
    for i in range(y_test.shape[0]):
        for j in range(n_topk+1):
            if j==n_topk:   
                if y_test[i] in m_class_sort[i,:n_topk]:
                    accuracy[j] += 1./y_test.shape[0]
                    for_tup[j] = 'Top-%d accuracy'%(j)
            else:
                if y_test[i] in m_class_sort[i,j:j+1]:
                    accuracy[j] += 1./y_test.shape[0]
                    for_tup[j] = 'PPV Rank %d'%(n_topk-j)
    p_res=list(zip(for_tup,accuracy))
    print('Performance {0}: {1}\n'.format(model.__class__.__name__,p_res))
                                                       
    return y_pred, p_res, class_sort, prob_sort


def best_topk_model(topk_models,n_topk,x_train,y_train,x_test,y_test,yencoder):
    model_prob_max, accuracy_max, best_model = 0., 0., ''
    for i in range(len(topk_models)):
        topk_performance(topk_models[i],n_topk,x_train,y_train,x_test,y_test,yencoder)
        if p_res[n_topk][1] > accuracy_max:
            accuracy_max = p_res[n_topk][1]
            model_prob_max = p_res
            best_model = topk_models[i].__class__.__name__
            ypred_max, p_res_max, class_sort_max, prob_sort_max = y_pred, p_res, class_sort, prob_sort       
    print('Best model: {0}, {1}\n'.format(best_model,model_prob_max))
    return ypred_max, p_res_max, class_sort_max, prob_sort_max
 
