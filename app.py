from sklearn.decomposition import PCA

from flask import Flask, render_template, jsonify

from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import tiktoken
enc = tiktoken.get_encoding('r50k_base')
from matplotlib import cm
from flask import request
import random



resids = pickle.load(open('10_token_resids.p', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            # Handle missing data
            raise Exception('Missing request data')

        layer = data.get('layer')
        component = data.get('component')
        diffs_on = data.get('diffs_on')
        basis_type = data.get('basis_type')
        
        if layer is not None:
            layer = int(layer)
        if component is not None:
            component = int(component)
        if diffs_on is not None:
            diffs_on = diffs_on.lower() == 'true'

        absoluteness_str = '' if diffs_on else 'absolute_'
        # X = np.array([resids[layer][i][0] for i in range(len(resids[layer]))])

        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        # # Perform PCA
        # pca = PCA(n_components=100)
        # X_pca = pca.fit_transform(X_scaled)

        component = pickle.load(open(f'10_token_{absoluteness_str}pca_layer_{layer}_component_{component_index}.p', 'rb'))
        if basis_type == 'Cartesian':
            component = np.zeros(component.shape)
            component[component_index] += 1
        X_pca = pickle.load(open(f'10_token_{absoluteness_str}x_pca_layer_{layer}.p', 'rb'))
        scaler = pickle.load(open(f'10_token_{absoluteness_str}scaler_layer_{layer}.p', 'rb'))
        explained_variance_ratio = pickle.load(open(f'10_token_{absoluteness_str}absolute_explained_variance_ratio_layer_{layer}.p', 'rb'))

        raw_projections = []
        resid_tups = []

        # print(list(resids_by_sentence[0].keys())[5])
        # print(resids_by_sentence[12][' Marilyn Monroe and James Dean are still icons for many'][0][0][0])

        sentences = random.sample(list(resids_by_sentence[layer].keys()), 50)

        variance_explained = explained_variance_ratio[component_index]

        print(component[0])

        for sentence in sentences:
            for resid, position in resids_by_sentence[layer][sentence]:
                resid_scaled = scaler.transform([resid])
                raw_projection = np.dot(component, resid_scaled.T)[0]
                raw_projections.append(raw_projection)
                resid_tups.append((resid, sentence, position))

        min_proj = np.min(np.array(raw_projections))
        max_proj = np.max(np.array(raw_projections))

        for resid_tup, raw_projection in zip(resid_tups, raw_projections):
            
            resid, sentence, position = resid_tup
            encoded_sentence = enc.encode(sentence)

            # Normalize projection to the range [0, 1]
            projection = (raw_projection - min_proj) / (max_proj - min_proj)

            # Convert projection to color
            rgb = cm.get_cmap('bwr')(projection)[:3]  # 'bwr' colormap ranges from blue (negative) through white (zero) to red (positive)
            color = "#{:02x}{:02x}{:02x}".format(*(int(x * 255) for x in rgb))

            assert position-1 >= 0
            word = enc.decode([encoded_sentence[position-1]])
            sentence_dicts.append({
                "sentence": sentence,
                "words": [{
                    "text": word,
                    "color": color,
                }],
            })

        # Merge dictionaries with the same sentence
        sentences = []
        for sentence_dict in sentence_dicts:
            # If this sentence is already in the list, add the word to it
            for existing_sentence in sentences:
                if existing_sentence["sentence"] == sentence_dict["sentence"]:
                    existing_sentence["words"].append(sentence_dict["words"][0])
                    break
            # If it's a new sentence, add it to the list
            else:
                sentences.append(sentence_dict)

        # print(sentence_dicts)
        # print(sentences)

        return render_template('index.html', sentences=sentences, layer=layer, component=component_index, diffs_on=diffs_on, variance_explained=f"{(variance_explained*100):.1f}" if basis_type == 'PCA' else '', basis_type=basis_type, min_proj=min_proj, max_proj=max_proj)

    else:
        return render_template('index.html', sentences=[], layer=0, component=0, diffs_on=False, variance_explained=0, basis_type='PCA', min_proj=None, max_proj=None)


@app.route('/process_sentence', methods=['POST'])
def process_sentence():
    from andrea_playing_around import get_resids_for_sentence

    sentence = request.form.get('sentence')
    layer = int(request.form.get('layer'))
    component_index = int(request.form.get('component'))    
    diffs_on = 'diffs_on' in request.form  # Returns True if checkbox was checked, False otherwise
    basis_type = request.form.get('basis_type')
    # ... process sentence ...

    diffs_on = 'diffs_on' in request.form  # Returns True if checkbox was checked, False otherwise
    resids = get_resids_for_sentence(sentence)

    absoluteness_str = '' if diffs_on else 'absolute_'

    component = pickle.load(open(f'10_token_{absoluteness_str}pca_layer_{layer}_component_{component_index}.p', 'rb'))
    if basis_type == 'Cartesian':
        component = np.zeros(component.shape)
        component[component_index] += 1
    X_pca = pickle.load(open(f'10_token_{absoluteness_str}x_pca_layer_{layer}.p', 'rb'))
    scaler = pickle.load(open(f'10_token_{absoluteness_str}scaler_layer_{layer}.p', 'rb'))

    resids_for_layer = resids[layer]
    for i, encoded_word in enumerate(enc.encode(sentence)):
        resid_for_position = resids_for_layer[0, i+1, :]


    # Assuming 'words' is a list of dictionaries, each containing 'text' and 'color' keys.
    print(layer)
    print(sentence)
    print(component_index)

    return jsonify({'words': words})

# for layer in resids:
#     X = np.array([resids[layer][i][0] for i in range(len(resids[layer]))]) 

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Perform PCA
#     pca = PCA(n_components=100)
#     X_pca = pca.fit_transform(X_scaled)

#     for resid, sentence, position in resids[layer][:60]:
#         resid_scaled = scaler.transform([resid])
#         projection = np.dot(pca.components_[0], resid_scaled.T)
#         encoded_sentence = enc.encode(sentence)

#         print(projection)
#         print(sentence)
#         assert position-1 >= 0
#         print(enc.decode([encoded_sentence[position-1]]))

#     print('')
#     print('')



# raise Exception()


if __name__ == '__main__':
    app.run(debug=True, port=5001)