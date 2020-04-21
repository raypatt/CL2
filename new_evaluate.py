def evaluate(filePath, model, vector_model, max_length, vocab) :
    import csv
    output_csv = open('output.csv', 'w')
    output_writer = csv.writer(output_csv)
    
    embedding_length = 100
    tests = get_test_file(filePath)
    embedded_tests = tests_to_embeddings(max_length, tests, vector_model, vocab)[2:]
    
    chunked_embedded_tests = chunks(embedded_tests, 10)
    chunked_tests = chunks(tests, 10) 
    
    
    tmp_tests = []
    for c in chunked_tests:
        tmp_tests.append(c)
        
    total_correct = 0
    total = 0 
    for chunk_idx, chunk in enumerate(chunked_embedded_tests):
        c1_list = []
        c2_list = []
        chunk_labels = []
        
        if len(chunk) == 10: 
            for test in chunk: 
                c1_list.append(test[0])
                c2_list.append(test[1])
                chunk_labels.append(test[2])
            
            c1_scores = model(torch.stack(c1_list))
            c2_scores = model(torch.stack(c2_list))
            
            
            ### Test if headlines are correctly paired
            ## This prints the first word embedding of the first test in both clusters
            ## It prints the cluster ID before the first word embedding of the first test of that cluster
            print (str(chunk_idx) + str((c1_list[0][0])))
            print (str(chunk_idx) + str((c2_list[0][0])))
            
            actual_labels = []
            for idx, c1_score in enumerate(c1_scores):
                label = 0
                
                
                if c1_score > c2_scores[idx]: 
                    label = 1
                if c1_score < c2_scores[idx]:
                    label = 2 
                actual_labels.append(label)
            
            for idx, i in enumerate(chunk_labels): 
                chunk_labels[idx] = int(i[0])
                
                
            for idx, predicted in enumerate(actual_labels): 
                output_writer.writerow([tmp_tests[chunk_idx][idx], predicted])
                actual = chunk_labels[idx]
                if predicted == actual: total_correct += 1
                total += 1
                
    print (total_correct/float(total))
                
        