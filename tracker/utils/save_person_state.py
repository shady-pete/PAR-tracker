import json
import re

def generate_output_file(dictionary):
    '''Generazione del file risultante avendo a disposizione quello utilizzato dalla classe <<LineTracker>>'''
    people_list= []
    for index_key in (dictionary):    
        person= dictionary[index_key]
        output_person= {}
        
        output_person['id']=            person['person_id']        
        output_person['gender']=        'male' if person['gender_probs']>= 0 else 'female'
        output_person['hat']=           bool(person['hat_probs']<=0)
        output_person['bag']=           bool(person['bag_probs']<=0)
        output_person['trajectory']=    [item[0] for item in person['trajectory']]
        
        people_list.append(output_person)
    
    
    people_list = sorted(people_list, key=lambda x: x['id'])

    people_output= {'people': people_list}
    return people_output

def save(dictionary, output_path="./output.json"):
    people_output= generate_output_file(dictionary)
    
    exit_file= json.dumps(people_output, indent=4)
    exit_file= re.sub(r"\n\s*(\d+)", r"\1", exit_file)
    exit_file= re.sub(r"\n\s*]", "]", exit_file)

    with open(output_path, 'w') as outfile:
        outfile.write(exit_file)