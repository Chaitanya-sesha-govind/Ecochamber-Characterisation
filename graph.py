
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import textwrap
from collections import defaultdict

# Constants
NUM_USERS = 36
LOCATIONS = ["India"]
MACRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
MICRO_TOPICS = ["Politics", "Sports", "Entertainment", "Technology"]
POLITICAL_LEANINGS = ["Government", "Opposition", "Neutral"]
ACTIVITY_LEVELS = ["Low Active", "Medium Active", "High Active"]
TWEET_POLARITY_VALUES = {"Government": 1, "Opposition": 0, "Neutral": 0.5}
ECHO_THRESHOLD = 0.3  # Clustering coefficient threshold

# Helper Functions
def assign_activity_level():
    return random.choice(ACTIVITY_LEVELS)

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

# Network Generation
def generate_scale_free_network(n_users, m=2, p=0.0):
    return nx.powerlaw_cluster_graph(n_users, m=m, p=p, seed=42)

def assign_political_leanings_by_degree(G):
    degrees = sorted(G.degree(), key=lambda x: -x[1])  # Sort by degree (high to low)
    leanings_cycle = [  "Government","Opposition","Neutral"]
    
    for i, (node, degree) in enumerate(degrees):
        leaning = leanings_cycle[i % 3]
        G.nodes[node]["Political Leaning"] = leaning
        G.nodes[node]["Degree"] = degree
    
    return G

# User Data Generation
def generate_user_data(G):
    data = []
    activity_levels_pool = ACTIVITY_LEVELS * (NUM_USERS // 3)
    random.shuffle(activity_levels_pool)

    for node in G.nodes():
        location = random.choice(LOCATIONS)
        political_leaning = G.nodes[node]["Political Leaning"]
        activity_level = activity_levels_pool.pop()
        degree = G.nodes[node]["Degree"]

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
            "User ID": f"U{node + 1}",
            "Location": location,
            "Political Leaning": political_leaning,
            "Activity Level": activity_level,
            "Degree": degree,
            "Macro Topic": macro_topic,
            "Micro Topic": micro_topic,
            "Number of Tweets": num_tweets,
            "Content Polarity": content_polarity,
            "Content Variance": content_variance,
            "δ-Partisan (δ=0.2)": delta_partisan_status
        })

    return pd.DataFrame(data)

def create_graph(G, user_data):
    for idx, row in user_data.iterrows():
        node = int(row["User ID"][1:]) - 1
        for col in user_data.columns:
            if col != "User ID":
                G.nodes[node][col] = row[col]
    return G

# Analysis Functions
def calculate_mixing_patterns(G):
    mixing = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            mixing[node] = 0
            continue
        node_leaning = G.nodes[node]['Political Leaning']
        total_score = 0
        for n in neighbors:
            neighbor_leaning = G.nodes[n]['Political Leaning']
            if neighbor_leaning == node_leaning:
                total_score += 1
            elif (node_leaning == "Government" and neighbor_leaning == "Opposition") or \
                 (node_leaning == "Opposition" and neighbor_leaning == "Government"):
                total_score -= 1
        mixing[node] = total_score / len(neighbors) if neighbors else 0
    return mixing

def calculate_ei_index(G):
    """Calculate the EI index for the entire network."""
    inter_group_edges = 0
    intra_group_edges = 0
    
    for u, v in G.edges():
        if G.nodes[u]['Political Leaning'] != G.nodes[v]['Political Leaning']:
            inter_group_edges += 1
        else:
            intra_group_edges += 1
    
    total_edges = inter_group_edges + intra_group_edges
    if total_edges == 0:
        return {'EI Index': 0, 'Inter-group Edges': 0, 'Intra-group Edges': 0}
    
    ei_index = (inter_group_edges - intra_group_edges) / total_edges
    return {'EI Index': ei_index, 'Inter-group Edges': inter_group_edges, 'Intra-group Edges': intra_group_edges}

def calculate_group_ei_indices(G):
    """Calculate EI indices for each political group."""
    groups = list(set(nx.get_node_attributes(G, 'Political Leaning').values()))
    ei_results = {}
    
    for group in groups:
        group_nodes = [n for n in G.nodes if G.nodes[n]['Political Leaning'] == group]
        other_nodes = [n for n in G.nodes if G.nodes[n]['Political Leaning'] != group]
        
        # Count edges within group and between group and others
        intra_edges = 0
        inter_edges = 0
        
        for node in group_nodes:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['Political Leaning'] == group:
                    intra_edges += 1
                else:
                    inter_edges += 1
        
        # Each edge is counted twice (once from each node)
        intra_edges = intra_edges // 2
        total_edges = intra_edges + inter_edges
        
        if total_edges == 0:
            ei_results[group] = {
                'EI Index': 0,
                'Inter-group Edges': 0,
                'Intra-group Edges': 0,
                'Total Edges': 0
            }
        else:
            ei_index = (inter_edges - intra_edges) / total_edges
            ei_results[group] = {
                'EI Index': ei_index,
                'Inter-group Edges': inter_edges,
                'Intra-group Edges': intra_edges,
                'Total Edges': total_edges
            }
    
    return ei_results

def calculate_node_ei_index(G):
    """Calculate EI index for each individual node."""
    node_ei = {}
    
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            node_ei[node] = 0
            continue
            
        same_group = 0
        diff_group = 0
        node_group = G.nodes[node]['Political Leaning']
        
        for neighbor in neighbors:
            if G.nodes[neighbor]['Political Leaning'] == node_group:
                same_group += 1
            else:
                diff_group += 1
                
        total = same_group + diff_group
        if total == 0:
            node_ei[node] = 0
        else:
            node_ei[node] = (diff_group - same_group) / total
    
    return node_ei

def identify_echo_chambers(G, threshold=ECHO_THRESHOLD):
    clustering = nx.clustering(G)
    nx.set_node_attributes(G, clustering, 'Clustering Coefficient')
    echo_chambers = {}
    for node in G.nodes():
        partisan = G.nodes[node]['δ-Partisan (δ=0.2)']
        if partisan and clustering[node] >= threshold:
            echo_chambers[node] = True
        else:
            echo_chambers[node] = False
    nx.set_node_attributes(G, echo_chambers, 'In Echo Chamber')
    return clustering, echo_chambers

def print_wrapped(text, width=120):
    wrapped_text = textwrap.fill(text, width=width)
    print(wrapped_text)

# Main Execution
G = generate_scale_free_network(NUM_USERS, m=2, p=0.0)
G = assign_political_leanings_by_degree(G)
df = generate_user_data(G)
G = create_graph(G, df)

mixing_patterns = calculate_mixing_patterns(G)
nx.set_node_attributes(G, mixing_patterns, 'Mixing Pattern')
df['Mixing Pattern'] = [mixing_patterns[i] for i in range(len(df))]

node_ei_indices = calculate_node_ei_index(G)
nx.set_node_attributes(G, node_ei_indices, 'Node EI Index')
df['Node EI Index'] = [node_ei_indices[i] for i in range(len(df))]

ei_results = calculate_ei_index(G)
group_ei_results = calculate_group_ei_indices(G)
clustering_coefficients, echo_chamber_flags = identify_echo_chambers(G)

df['Clustering Coefficient'] = df.index.map(clustering_coefficients)
df['In Echo Chamber'] = df.index.map(echo_chamber_flags)

# Visualization
plt.figure(figsize=(20, 15))
color_map = {"Government": "red", "Opposition": "blue", "Neutral": "green"}
colors = [color_map[G.nodes[node]['Political Leaning']] for node in G.nodes]
pos = nx.spring_layout(G, seed=42)

degrees = [G.nodes[node]['Degree'] for node in G.nodes]
min_degree, max_degree = min(degrees), max(degrees)
sizes = [300 + 1200 * (d - min_degree)/(max_degree - min_degree) for d in degrees]

nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.9)
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_labels(G, pos, font_size=8)

for node in G.nodes:
    x, y = pos[node]
    info = (
            f"D:{G.nodes[node]['Degree']}\n"
            f"M:{G.nodes[node]['Mixing Pattern']:.2f}\n"
            f"P:{G.nodes[node]['Content Polarity']:.2f}\n"
            f"δ:{G.nodes[node]['δ-Partisan (δ=0.2)'][:1]}\n"
            f"C:{G.nodes[node]['Clustering Coefficient']:.2f}\n"
            f"EI:{G.nodes[node]['Node EI Index']:.2f}"
            )
    plt.text(x, y + 0.07, info, ha='center', va='center', fontsize=7,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

for leaning, color in color_map.items():
    plt.scatter([], [], c=color, label=f"{leaning}")
plt.legend(title="Political Leaning")
plt.title("Node size = Degree | M = Mixing pattern | P = Content polarity | δ = Partisan status | C = Clustering | EI = Node EI Index")
plt.axis('off')

# Console Output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)






plt.tight_layout()


print("\n" + "="*120)
print_wrapped("COMPLETE USER DATA")
print("="*120)
print(df.sort_values('Degree', ascending=False))

print("\n" + "="*120)
print_wrapped("ECHO CHAMBER USERS (Clustering ≥ 0.3 AND Partisan)")
print("="*120)
print(df[df['In Echo Chamber'] == True][['User ID', 'Political Leaning', 'δ-Partisan (δ=0.2)', 'Clustering Coefficient']])

print("\n" + "="*120)
print_wrapped("NODE EI INDEX ANALYSIS")
print("="*120)
print(df[['User ID', 'Political Leaning', 'Node EI Index']].sort_values('Node EI Index', ascending=False))

print("\n" + "="*120)
print_wrapped("NODE EI INDEX INTERPRETATION")
print("="*120)
print_wrapped("EI Index ranges from -1 to 1:")
print_wrapped("- Values close to -1 indicate strong homophily (mostly within-group connections)")
print_wrapped("- Values close to 1 indicate strong heterophily (mostly between-group connections)")
print_wrapped("- Values around 0 indicate balanced connections")

print("\n" + "="*120)
print_wrapped("EI INDEX ANALYSIS")
print("="*120)

print("\nOverall Network:")
for key, value in ei_results.items():
    print_wrapped(f"{key}: {value}")

print("\nGroup-Specific Analysis:")
for group, results in group_ei_results.items():
    print(f"\n{group} Group:")
    for key, value in results.items():
        print_wrapped(f"  {key}: {value}")
    interpretation = "Homophily" if results['EI Index'] < 0 else "Heterophily"
    strength = "strong" if abs(results['EI Index']) > 0.5 else "moderate" if abs(results['EI Index']) > 0.2 else "weak"
    print_wrapped(f"  Interpretation: {interpretation} ({strength})")

# Create histograms for activity levels and political leanings
plt.figure(figsize=(15, 10))

# High Active users
plt.subplot(3, 1, 1)
high_active = df[df['Activity Level'] == 'High Active']
high_active_counts = high_active['Political Leaning'].value_counts()
colors = [color_map[leaning] for leaning in high_active_counts.index]
high_active_counts.plot(kind='bar', color=colors)
plt.title('High Active Users by Political Leaning')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Medium Active users
plt.subplot(3, 1, 2)
medium_active = df[df['Activity Level'] == 'Medium Active']
medium_active_counts = medium_active['Political Leaning'].value_counts()
colors = [color_map[leaning] for leaning in medium_active_counts.index]
medium_active_counts.plot(kind='bar', color=colors)
plt.title('Medium Active Users by Political Leaning')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Low Active users
plt.subplot(3, 1, 3)
low_active = df[df['Activity Level'] == 'Low Active']
low_active_counts = low_active['Political Leaning'].value_counts()
colors = [color_map[leaning] for leaning in low_active_counts.index]
low_active_counts.plot(kind='bar', color=colors)
plt.title('Low Active Users by Political Leaning')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.tight_layout()



# Show both figures
plt.figure(1)  # The network graph
plt.show()

plt.figure(2)  # The histograms
plt.show()