import pandas as pd
import random
import numpy as np
import textwrap
import networkx as nx
import matplotlib.pyplot as plt


NUM_USERS = 36
LOCATIONS = ["India"]
MACRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
MICRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
POLITICAL_LEANINGS = ["Government", "Opposition", "Neutral"]
ACTIVITY_LEVELS = ["Low Active", "Medium Active", "High Active"]
TWEET_POLARITY_VALUES = {"Government": 1, "Opposition": 0, "Neutral": 0.5}

def assign_activity_level():
    return random.choice(ACTIVITY_LEVELS)

def assign_political_leaning():
    return random.choice(POLITICAL_LEANINGS)

def generate_tweet_polarities(political_leaning, num_tweets):
    if political_leaning == "Government":
        gov_percent = random.uniform(0.7, 1.0)
        neutral_percent = random.uniform(0, 1 - gov_percent)
        opp_percent = 1 - gov_percent - neutral_percent
    elif political_leaning == "Opposition":
        opp_percent = random.uniform(0.7, 1.0)
        neutral_percent = random.uniform(0, 1 - opp_percent)
        gov_percent = 1 - opp_percent - neutral_percent
    else:  
        neutral_percent = random.uniform(0.6, 1.0)
        gov_percent = random.uniform(0, 1 - neutral_percent)
        opp_percent = 1 - neutral_percent - gov_percent
    
    num_gov = int(round(gov_percent * num_tweets))
    num_opp = int(round(opp_percent * num_tweets))
    num_neutral = num_tweets - num_gov - num_opp

    tweet_polarities = (
        [TWEET_POLARITY_VALUES["Government"]] * num_gov +
        [TWEET_POLARITY_VALUES["Opposition"]] * num_opp +
        [TWEET_POLARITY_VALUES["Neutral"]] * num_neutral
    )
    random.shuffle(tweet_polarities)  
    return tweet_polarities

def calculate_content_polarity(tweet_polarities):
    return np.mean(tweet_polarities)

def calculate_content_variance(tweet_polarities):
    return np.var(tweet_polarities)

def is_delta_partisan(content_polarity, delta=0.2):
    if content_polarity <= delta:
        return "Biased (Opposition)"
    elif content_polarity >= 1 - delta:
        return "Biased (Government)"
    else:
        return "Not Biased"

# Generate user data
data = []
activity_levels_pool = ACTIVITY_LEVELS * (NUM_USERS // 3) 
political_leanings_pool = POLITICAL_LEANINGS * (NUM_USERS // 3)  
random.shuffle(activity_levels_pool)
random.shuffle(political_leanings_pool)

for user_id in range(1, NUM_USERS + 1):
    location = random.choice(LOCATIONS)
    political_leaning = political_leanings_pool.pop()
    activity_level = activity_levels_pool.pop()
    
    if activity_level == "Low Active":
        num_tweets = random.randint(1, 5)
    elif activity_level == "Medium Active":
        num_tweets = random.randint(6, 10)
    else:
        num_tweets = random.randint(11, 20) 
    
    tweet_polarities = generate_tweet_polarities(political_leaning, num_tweets)
    content_polarity = calculate_content_polarity(tweet_polarities)
    content_variance = calculate_content_variance(tweet_polarities)
    delta_partisan_status = is_delta_partisan(content_polarity)
    
    macro_topic = random.choice(MACRO_TOPICS)
    micro_topic = random.choice([t for t in MICRO_TOPICS if t != macro_topic])
    
    data.append({
        "User ID": f"U{user_id}",
        "Location": location,
        "Political Leaning": political_leaning,
        "Activity Level": activity_level,
        "Macro Topic": macro_topic,
        "Micro Topic": micro_topic,
        "Number of Tweets": num_tweets,
        "Content Polarity": content_polarity,
        "Content Variance": content_variance,
        "δ-Partisan (δ=0.2)": delta_partisan_status
    })

df = pd.DataFrame(data)


def generate_random_adjacency(n_users, connection_chance=0.15):
    adj = np.random.random((n_users, n_users))
    adj = (adj < connection_chance).astype(int)
    np.fill_diagonal(adj, 0)  
    adj = np.triu(adj) + np.triu(adj).T 
    return adj


def create_graph(adj_matrix, user_data):
    G = nx.from_numpy_array(adj_matrix)
    for idx, row in user_data.iterrows():
        for col in user_data.columns:
            G.nodes[idx][col] = row[col]
    return G

def calculate_node_ei_index(G):
    node_ei = {}
    groups = {n: G.nodes[n]['Political Leaning'] for n in G.nodes}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            node_ei[node] = 0
            continue
        same_group = 0
        diff_group = 0
        for neighbor in neighbors:
            if groups[node] == groups[neighbor]:
                same_group += 1
            else:
                diff_group += 1
        total = same_group + diff_group
        if total == 0:
            node_ei[node] = 0
        else:
            node_ei[node] = (diff_group - same_group) / total
    return node_ei


def calculate_mixing_patterns(G):
    mixing = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            mixing[node] = 0
            continue
        same = sum(1 for n in neighbors if G.nodes[n]['Political Leaning'] == G.nodes[node]['Political Leaning'])
        mixing[node] = same / len(neighbors)
    return mixing


def calculate_ei_index(G):
    inter_group_edges = 0
    intra_group_edges = 0
    for edge in G.edges():
        node1 = edge[0]
        node2 = edge[1]
        if G.nodes[node1]['Political Leaning'] != G.nodes[node2]['Political Leaning']:
            inter_group_edges += 1
        else:
            intra_group_edges += 1
    if (inter_group_edges + intra_group_edges) == 0:
        return {'EI Index': 0, 'Inter-group Edges': 0, 'Intra-group Edges': 0}
    ei_index = (inter_group_edges - intra_group_edges) / (inter_group_edges + intra_group_edges)
    return {'EI Index': ei_index, 'Inter-group Edges': inter_group_edges, 'Intra-group Edges': intra_group_edges}

# Calculate EI Index for each group
def calculate_group_ei_indices(G):
    groups = list(set(nx.get_node_attributes(G, 'Political Leaning').values()))
    ei_results = {}
    for group in groups:
        inter_group_edges = 0
        intra_group_edges = 0
        for u, v in G.edges():
            u_group = G.nodes[u]['Political Leaning']
            v_group = G.nodes[v]['Political Leaning']
            if u_group == group and v_group == group:
                intra_group_edges += 1
            elif (u_group == group and v_group != group) or (u_group != group and v_group == group):
                inter_group_edges += 1
        total_edges = inter_group_edges + intra_group_edges
        if total_edges == 0:
            ei_index = 0
        else:
            ei_index = (inter_group_edges - intra_group_edges) / total_edges
        ei_results[group] = {
            'EI Index': ei_index,
            'Inter-group Edges': inter_group_edges,
            'Intra-group Edges': intra_group_edges,
            'Total Edges': total_edges
        }
    return ei_results


def calculate_clustering(G):
    return nx.clustering(G)

def print_wrapped(text, width=120):
    wrapped_text = textwrap.fill(text, width=width)
    print(wrapped_text)


adj_matrix = generate_random_adjacency(len(df))
G = create_graph(adj_matrix, df)
mixing_patterns = calculate_mixing_patterns(G)
clustering_coeffs = calculate_clustering(G)


nx.set_node_attributes(G, mixing_patterns, 'Mixing Pattern')
nx.set_node_attributes(G, clustering_coeffs, 'Clustering Coefficient')
df['Mixing Pattern'] = [mixing_patterns[i] for i in range(len(df))]
df['Clustering Coefficient'] = [clustering_coeffs[i] for i in range(len(df))]
node_ei_indices = calculate_node_ei_index(G)
nx.set_node_attributes(G, node_ei_indices, 'Node EI Index')
df['Node EI Index'] = [node_ei_indices[i] for i in range(len(df))]


ei_results = calculate_ei_index(G)
group_ei_results = calculate_group_ei_indices(G)




plt.figure(figsize=(14, 10))
color_map = {"Government": "red", "Opposition": "blue", "Neutral": "green"}
colors = [color_map[G.nodes[node]['Political Leaning']] for node in G.nodes]
pos = nx.spring_layout(G, seed=42)


size_map = {"Low Active": 200, "Medium Active": 400, "High Active": 600}
sizes = [size_map[G.nodes[node]['Activity Level']] for node in G.nodes]

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=8)


for node in G.nodes:
    x, y = pos[node]
    info = (
        f"M:{G.nodes[node]['Mixing Pattern']:.2f}\n"
        f"Variance :{G.nodes[node]['Content Variance']:.2f}\n"
        f"δ:{G.nodes[node]['δ-Partisan (δ=0.2)'][:1]}\n"
        f"CC:{G.nodes[node]['Clustering Coefficient']:.2f}\n"
        f"EI:{G.nodes[node]['Node EI Index']:.2f}"
        )
    plt.text(x, y+0.07, info, ha='center', va='center', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Legend
for leaning, color in color_map.items():
    plt.scatter([], [], c=color, label=f"{leaning}")
plt.legend(title="Political Leaning")

plt.title("Random Social Network Graph\n"
          "Node size = Activity Level | "
          "M = Mixing pattern | C = Content polarity | δ = Partisan status | CC = Clustering Coefficient | EI = Node EI Index")
plt.axis('off')

# Print results
print("="*80)
print("COMPLETE USER DATA")
print("="*80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df)

print("\n" + "="*80)
print("NETWORK METRICS")
print("="*80)
print("\nClustering Coefficients:")
print(df['Clustering Coefficient'].describe())

print("\n" + "="*120)
print_wrapped("NODE EI INDEX ANALYSIS")
print("="*120)
print(df[['User ID', 'Political Leaning', 'Node EI Index']].sort_values('Node EI Index', ascending=False))
print("\n" + "="*80)

print("EI INDEX ANALYSIS")
print("="*80)
print("\nOverall Network:")
for key, value in ei_results.items():
    print(f"{key}: {value}")

print("\nGroup-Specific Analysis:")
for group, results in group_ei_results.items():
    print(f"\n{group} Group:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    interpretation = "Homophily" if results['EI Index'] < 0 else "Heterophily"
    strength = "strong" if abs(results['EI Index']) > 0.5 else "moderate" if abs(results['EI Index']) > 0.2 else "weak"
    print(f"  Interpretation: {interpretation} ({strength})")

plt.tight_layout()
plt.show()

