use rand::seq::SliceRandom;
use std::{
    cmp::{max, min},
    collections::{BinaryHeap, HashMap, HashSet},
};

use crate::{
    automaton::{AppRegex, BaseAutomaton, State, Transition, NFA},
    conf_automaton::PCPConf,
    pcp::{PCPConfig, PCPDir, PCP},
    suffix_tree::{NaiveSuffixTree, SuffixTree},
    union_find::UnionFind,
};
use itertools::Itertools;
use rand::random;

#[derive(Clone, Debug)]
enum DependsOn {
    Nexts(Vec<NodeId>),
    Abstract((Option<Vec<HashSet<State>>>, NodeId)),
}

impl DependsOn {
    fn get_dependent_nodes(&self) -> Vec<NodeId> {
        match self {
            DependsOn::Nexts(nexts) => nexts.clone(),
            DependsOn::Abstract((_, id)) => vec![*id],
        }
    }
}

fn to_substring_nfa(nfa: &mut NFA<char>) {
    nfa.add_transition(Transition{
        from:nfa.start.clone(),
        to:nfa.start.clone(),
        label:Some('0'),
    });
    nfa.add_transition(Transition{
        from:nfa.start.clone(),
        to:nfa.start.clone(),
        label:Some('1'),
    });
    for ac in nfa.accept.clone() {
        nfa.add_transition(Transition{
            from:ac.clone(),
            to:ac.clone(),
            label:Some('0'),
        });
        nfa.add_transition(Transition{
            from:ac.clone(),
            to:ac.clone(),
            label:Some('1'),
        });
    }
}

fn nexts(seq: &PCPConf, pcp: &PCP) -> Vec<PCPConf> {
    seq.apply_pcp(pcp)
}

fn prevs(seq: &PCPConfig, pcp: &PCP) -> Vec<PCPConfig> {
    seq.co()
        .apply_pcp(&pcp.co())
        .into_iter()
        .map(|c| c.reverse())
        .collect_vec()
}

#[derive(Clone, Debug)]
struct Node {
    id: NodeId,
    bad: bool,
    dirty: bool,
    seq: PCPConf,
    depends_on: Option<DependsOn>,
    referenced_by: HashSet<NodeId>,
}

impl Node {
    fn new(id: NodeId, seq: PCPConf) -> Self {
        Self {
            id,
            bad: false,
            dirty: false,
            seq: PCPConf {
                dir: seq.dir,
                conf: seq.conf.reduce_size(),
                exact: seq.exact,
            },
            depends_on: None,
            referenced_by: HashSet::new(),
        }
    }

    fn get_dependent_nodes(&self) -> Vec<NodeId> {
        match &self.depends_on {
            Some(dep) => dep.get_dependent_nodes(),
            None => vec![],
        }
    }
}

struct BadConfigStore {
    bad_up: NaiveSuffixTree,
    bad_dn: NaiveSuffixTree,
}
impl BadConfigStore {
    fn new() -> BadConfigStore {
        Self {
            bad_up: NaiveSuffixTree::new(),
            bad_dn: NaiveSuffixTree::new(),
        }
    }

    fn add_config(&mut self, conf: &PCPConfig) {
        match conf.dir {
            PCPDir::UP => {
                self.bad_up.insert(conf.seq.as_str());
            }
            PCPDir::DN => {
                self.bad_dn.insert(conf.seq.as_str());
            }
        }
    }

    fn contains_subset_of(&self, seq: &PCPConf) -> bool {
        match seq.dir {
            PCPDir::UP => self
                .bad_up
                .strs
                .iter()
                .any(|s| seq.conf.accept(&s.chars().collect_vec())),
            PCPDir::DN => self
                .bad_dn
                .strs
                .iter()
                .any(|s| seq.conf.accept(&s.chars().collect_vec())),
        }
    }
}

const MAX_SUBSTR_LEN: u32 = 40;

struct Graph {
    pcp: PCP,
    bad: BadConfigStore,

    nodes: Vec<Node>,

    reachable_frontier: BinaryHeap<(i32, NodeId)>, // (priority, id)
    reachable_dirties: HashSet<NodeId>,
    reachable: HashSet<NodeId>,

    mid_to_node: HashMap<PCPDir, HashMap<String, NodeId>>,
    exact_to_node: HashMap<PCPDir, HashMap<String, NodeId>>,

    starts: HashSet<NodeId>,
}

fn substrings(s: &str, min_len: usize, max_len: usize) -> Vec<String> {
    let mut ret = HashSet::<String>::new();

    for len in min_len..=max_len {
        for i in 0..=s.len() - len {
            ret.insert(s[i..i + len].to_string());
        }
    }

    ret.into_iter().collect_vec()
}

impl<'a> Graph {
    fn new(pcp: PCP) -> Self {
        let mut g = Self {
            pcp: pcp.clone(),
            bad: BadConfigStore::new(),
            nodes: vec![],
            reachable_frontier: BinaryHeap::new(),
            reachable_dirties: HashSet::new(),
            reachable: HashSet::new(),
            mid_to_node: HashMap::new(),
            exact_to_node: HashMap::new(),
            starts: HashSet::new(),
        };
        g.mid_to_node.insert(PCPDir::UP, HashMap::new());
        g.mid_to_node.insert(PCPDir::DN, HashMap::new());
        g.exact_to_node.insert(PCPDir::UP, HashMap::new());
        g.exact_to_node.insert(PCPDir::DN, HashMap::new());

        let confs = pcp.co().enumerate_configurations(100000);
        println!("max_len: {:?}", confs.iter().map(|c| c.seq.len()).max());
        g.add_bad(&PCPConfig {
            seq: "".to_string(),
            dir: PCPDir::DN,
        });
        g.add_bad(&PCPConfig {
            seq: "".to_string(),
            dir: PCPDir::UP,
        });
        for conf in confs.into_iter() {
            g.add_bad(&conf.reverse());
        }

        for conf in pcp.get_init_config().iter() {
            g.seq_to_node(PCPConf::from_exact(conf));
        }
        g.starts = g.nodes.iter().map(|n| n.id).collect();

        // g.seq_to_node(PCPConf {
        //     dir: PCPDir::UP,
        //     conf: NFA::from_regex(&AppRegex::parse("11(11)*")),
        //     exact: None,
        // });

        g.recompute_start_component();
        println!(
            "finished init: {:?} {:?}",
            g.bad.bad_dn.size(),
            g.bad.bad_up.size()
        );
        g
    }

    fn is_starts_bad(&self) -> bool {
        self.starts.iter().any(|id| self.nodes[*id].bad)
    }

    fn find_abstraction_node_for(&self, node: &Node) -> Option<NodeId> {
        for other_id in self.reachable.iter() {
            let other = self.get_node(*other_id).unwrap();
            if other.seq.exact.is_none()
                && node.seq.dir == other.seq.dir
                && !other.bad
                && node.id != other.id
                && other.seq.conf.includes(&node.seq.conf)
                && !node.seq.conf.includes(&other.seq.conf)
            {
                return Some(other.id);
            }
        }
        None
    }

    fn find_concretize_node_for(&self, node: &Node) -> Vec<NodeId> {
        self.reachable.iter().filter(|other_id| {
            let other = self.get_node(**other_id).unwrap();
            return  node.seq.dir == other.seq.dir
                && !other.bad
                && node.id != other.id
                && node.seq.conf.includes(&other.seq.conf)
                && !other.seq.conf.includes(&node.seq.conf)
        }).cloned().collect_vec()
    }

    fn seq_to_node(&'a mut self, seq: PCPConf) -> &'a Node {
        let id = if let Some(e) = self
            .nodes
            .iter()
            .find(|n| n.seq.dir == seq.dir && n.seq.conf.is_equal(&seq.conf))
        {
            e.id
        } else {
            let id = self.nodes.len();
            let node = Node::new(id, seq);
            self.add_node(node);
            id
        };
        self.get_node(id).unwrap()
    }

    fn get_bad_path(&self, id: NodeId) -> Vec<String> {
        assert!(self.nodes[id].bad);

        let mut ret = vec![];
        let mut cur = id;
        while self.nodes[cur].bad && self.nodes[cur].depends_on.is_some() {
            ret.push(format!("{:?}", self.nodes[cur].seq));
            let next = self.nodes[cur]
                .get_dependent_nodes()
                .into_iter()
                .filter(|n| self.nodes[*n].bad)
                .next();
            cur = next.unwrap();
        }
        ret.push(format!("{:?}", self.nodes[cur].seq));
        ret
    }

    fn is_reachable(&self, id: NodeId) -> bool {
        self.reachable.contains(&id)
    }

    fn add_nodes_to_start_component(&mut self, root_id: NodeId) {
        if self.reachable.contains(&root_id) {
            return;
        }
        self.reachable.insert(root_id);

        let (is_dirty, is_frontier) = {
            let node = self.get_node(root_id).unwrap();
            (node.dirty, node.depends_on.is_none())
        };

        if is_dirty {
            self.reachable_dirties.insert(root_id);
        }
        if is_frontier {
            self.reachable_frontier.push((
                -(self.nodes[root_id].seq.conf.states.len() as i32
                    + if self.nodes[root_id].seq.exact.is_some() {
                        1000
                    } else {
                        0
                    }),
                root_id,
            ));
        }

        for dep in self.get_node(root_id).unwrap().get_dependent_nodes() {
            self.add_nodes_to_start_component(dep);
        }
    }

    fn recompute_start_component(&mut self) {
        self.reachable.clear();
        self.reachable_frontier.clear();
        self.reachable_dirties.clear();
        let starts = self.starts.clone();
        for s in starts {
            self.add_nodes_to_start_component(s);
        }
    }

    fn one_step_concretize(&mut self, node_id: NodeId) {
        let new_dep = if let Some(abs) = self.find_abstraction_node_for(&self.nodes[node_id]) {
            DependsOn::Abstract((None, abs))
        } else {
            match &self.nodes[node_id].depends_on {
                None => self.create_abstraction(&self.nodes[node_id].seq.clone()),
                Some(DependsOn::Nexts(ref nexts)) => {
                    println!("warn: concretizing a node that already concrete");
                    self.nodes[node_id].depends_on.clone().unwrap()
                }
                Some(DependsOn::Abstract((None, id))) => {
                    self.create_abstraction(&self.nodes[node_id].seq.clone())
                }
                Some(DependsOn::Abstract((Some(ds), id))) => {
                    let mut ds: Vec<HashSet<String>> = ds.clone();
                    loop {
                        if ds.len() == 0 {
                            break DependsOn::Nexts(
                                nexts(&self.nodes[node_id].seq, &self.pcp)
                                    .into_iter()
                                    .map(|seq| self.seq_to_node(seq).id)
                                    .collect(),
                            );
                        }

                        let random_idx = random::<usize>() % ds.len();
                        let mut removed_set = ds.remove(random_idx);
                        removed_set.remove(&removed_set.iter().next().unwrap().clone());
                        ds.push(removed_set);
                        ds.retain(|s| s.len() > 1);

                        let mut nfa = self.nodes[node_id].seq.conf.clone();
                        
                        to_substring_nfa(&mut nfa);
                        if self.is_contains_bad(&PCPConf {
                            dir: self.nodes[node_id].seq.dir,
                            conf: nfa.clone(),
                            exact: None,
                        }) {
                            nfa = self.nodes[node_id].seq.conf.clone();
                        }

                        let new_nfa = nfa
                            .merge_states(ds.iter().map(|s| s.iter().collect()).collect_vec());
                        if !self.nodes[node_id].seq.conf.is_equal(&new_nfa) {
                            break DependsOn::Abstract((
                                Some(ds),
                                self.seq_to_node(PCPConf {
                                    dir: self.nodes[node_id].seq.dir,
                                    conf: new_nfa,
                                    exact: None,
                                })
                                .id,
                            ));
                        }
                    }
                }
            }
        };
        self.set_node_dependency(node_id, new_dep);
    }
    fn set_node_is_dirty(&mut self, id: NodeId) {
        self.nodes[id].dirty = true;
        if self.is_reachable(id) {
            self.reachable_dirties.insert(id);
        }
    }
    fn process_dirty(&mut self, id: NodeId) {
        self.reachable_dirties.remove(&id);
        self.nodes.get_mut(id).unwrap().dirty = false;
        let node = self.get_node(id).unwrap();
        //println!("{:?} -> {:?}", id, node.depends_on);

        match &node.depends_on {
            Some(DependsOn::Nexts(_)) => {
                self.notify_node_is_bad(node.id);
            }
            Some(DependsOn::Abstract((_, _))) => {
                self.one_step_concretize(id);
            }
            None => todo!(),
        }
    }
    fn add_bad(&mut self, conf: &PCPConfig) {
        self.bad.add_config(conf);
    }
    fn is_contains_bad(&self, seq: &PCPConf) -> bool {
        self.bad.contains_subset_of(seq)
    }
    fn add_node(&mut self, node: Node) {
        if self.get_node(node.id).is_some() {
            return;
        }
        self.nodes.push(node.clone());

        if node.depends_on.is_none() {
            self.add_nodes_to_start_component(node.id);
        }
        if self.is_contains_bad(&node.seq) {
            self.notify_node_is_bad(node.id);
        } else {
            let concs = self.find_concretize_node_for(&node);
            println!("concs {:?}", concs.len());
            for conc in concs {
                // if let Some(DependsOn::Abstract((_, id))) = &self.nodes[conc].depends_on {
                //     if self.nodes[*id].seq.conf.states.len() <= node.seq.conf.states.len() {
                //         println!("skip {:?} {:?}", node.id, id);
                //         continue;
                //     }
                // }
                self.set_node_dependency(conc, DependsOn::Abstract((None, node.id)));
            }
        }
    }
    fn pop_dirty_node(&mut self) -> Option<NodeId> {
        if self.reachable_dirties.len() == 0 {
            return None;
        }
        let id = self.reachable_dirties.iter().next().unwrap().clone();
        self.reachable_dirties.remove(&id);
        Some(id)
    }
    fn pop_frontier_node(&'a mut self) -> Option<&'a Node> {
        loop {
            let cur = self.reachable_frontier.pop();
            if cur.is_none() {
                return None;
            }
            let (_, id) = cur.unwrap();
            if self.nodes[id].depends_on.is_none() {
                return self.get_node(id);
            }
        }
    }
    fn get_node(&'a self, id: NodeId) -> Option<&'a Node> {
        self.nodes.get(id)
    }
    fn set_node_dependency(&mut self, id: NodeId, depends_on: DependsOn) {
        for dep in self.nodes[id].get_dependent_nodes() {
            self.nodes[dep].referenced_by.remove(&id);
        }
        let mut is_ditry = false;

        self.nodes[id].depends_on = Some(depends_on);

        for dep in self.nodes[id].get_dependent_nodes() {
            self.nodes[dep].referenced_by.insert(id);
            is_ditry |= self.nodes[dep].bad;
            self.add_nodes_to_start_component(dep);
        }

        if is_ditry {
            self.set_node_is_dirty(id);
        }
    }

    fn create_abstraction(&mut self, seq: &PCPConf) -> DependsOn {
        let mut shuffled = seq.conf.states.iter().tuple_combinations().collect_vec();
        shuffled.shuffle(&mut rand::thread_rng());
        println!("p {:?}", shuffled.len());

        let mut ds = UnionFind::<State>::new();
        let mut nfa = seq.conf.clone();

        to_substring_nfa(&mut nfa);

        if self.is_contains_bad(&PCPConf {
            dir: seq.dir,
            conf: nfa.clone(),
            exact: None,
        }) {
            nfa = seq.conf.clone()
            // nfa = seq.conf.clone() にするより諦めた方が良さそう 1 false など
            // return DependsOn::Nexts(
            //     nexts(seq, &self.pcp)
            //         .into_iter()
            //         .map(|seq| self.seq_to_node(seq).id)
            //         .collect(),
            // );
        }

        let mut bad_merges: Vec<(&State, &State)> = vec![];
        for (s, t) in shuffled.iter() {
            if !ds.connected(s, t) && nfa.states.contains(*s) && nfa.states.contains(*t) {
                if bad_merges.iter().any(|(a, b)| ds.connected(s, a) && ds.connected(t, b)) {
                    continue;
                }

                let try_nfa = nfa.merge_states(vec![[*s, *t].into_iter().collect()]);
                let bad = self.is_contains_bad(&PCPConf {
                    dir: seq.dir,
                    conf: try_nfa.clone(),
                    exact: None,
                });
                //println!("p2 {:?} ({:?} {:?}) {:?} -> {:?}", bad, s, t, nfa.states.len(), try_nfa.states.len());
                if !bad {
                    nfa = try_nfa;
                    ds.merge(s, t);
                } else {
                    bad_merges.push((*s, *t));
                }
            }
        }

        if nfa.is_equal(&seq.conf) {
            return DependsOn::Nexts(
                nexts(seq, &self.pcp)
                    .into_iter()
                    .map(|seq| self.seq_to_node(seq).id)
                    .collect(),
            );
        }

        let mut hm = ds.to_hashmaps();
        hm.retain(|s| s.len() > 1);
        println!("p");
        return DependsOn::Abstract((
            Some(hm),
            self.seq_to_node(PCPConf {
                dir: seq.dir,
                conf: nfa,
                exact: None,
            })
            .id,
        ));
    }

    fn notify_node_is_bad(&mut self, id: NodeId) {
        if self.nodes[id].bad {
            return;
        }
        self.nodes[id].bad = true;
        if self.is_reachable(id) {
            for ref_by in self.nodes[id].referenced_by.clone().into_iter() {
                self.set_node_is_dirty(ref_by);
            }
        }
    }

    fn get_invariant(&mut self) -> Vec<Node> {
        self.recompute_start_component();
        self.reachable.iter().for_each(|id| {
            println!(
                "{:?} -> {:?}",
                self.nodes[*id].seq, self.nodes[*id].depends_on
            );
        });
        assert!(self
            .reachable
            .iter()
            .all(|&id| !self.nodes[id].bad && self.nodes[id].depends_on.is_some()));

        self.reachable
            .iter()
            .map(|id| self.nodes[*id].clone())
            .collect()
    }

    fn get_graph_dot(&self) -> String {
        let mut ret = "digraph G {\n".to_string();
        for id in self.reachable.iter() {
            let node = &self.nodes[*id];
            ret += &format!(
                "{} [label=\"{:?},{}\", shape=\"{}\", style=\"filled\", fillcolor=\"{}\"]\n",
                node.id,
                node.seq.dir,
                match &node.seq.exact {
                    Some(e) => e.to_string(),
                    None => node.seq.conf.states.len().to_string(),
                },
                match node.seq.exact {
                    Some(_) => "box",
                    None => "ellipse",
                },
                if node.bad {
                    "red"
                } else if node.dirty {
                    "gray"
                } else if node.depends_on.is_none() {
                    "yellow"
                } else {
                    "white"
                }
            );
            let is_abstract_dep = if let Some(DependsOn::Abstract((_, id))) = &node.depends_on {
                true
            } else {
                false
            };
            for dep in node.get_dependent_nodes() {
                ret += &format!(
                    "{} -> {} [style=\"{}\",label=\"{}\"]\n",
                    node.id,
                    dep,
                    if is_abstract_dep { "dotted" } else { "solid" },
                    if let Some(DependsOn::Abstract((a, _))) = &node.depends_on {
                        format!("{:?}", a).replace('\"', "")
                    } else {
                        "".to_string()
                    }
                );
            }
        }
        for start in self.starts.iter() {
            ret += &format!("start -> {} [style=\"solid\"]\n", start);
            ret += &format!("start [label=\"\", shape=\"point\"]\n");
        }
        ret += "}\n";
        ret
    }

    fn print_graph_dot(&self) {
        let dot = self.get_graph_dot();
        std::fs::write("graph.dot", dot).unwrap();
        let output = std::process::Command::new("dot")
            .arg("-Tpng")
            .arg("graph.dot")
            .arg("-o")
            .arg("graph.png")
            .output()
            .unwrap();
        println!("{:?}", output);
    }

    fn step(&mut self) -> bool {
        println!("step");
        if let Some(dirty_node) = self.pop_dirty_node() {
            self.process_dirty(dirty_node);
            return false;
        }
        //self.print_graph_dot();

        self.recompute_start_component();
        if let Some(id) = { self.pop_frontier_node().map(|n| n.id) } {
            println!("o");
            self.one_step_concretize(id);
            println!("o2");
        }
        println!(
            "reachable: {:?} frontier: {:?} dirties: {:?}",
            self.reachable.len(),
            self.reachable_frontier.len(),
            self.reachable_dirties.len()
        );
        return (self.reachable_dirties.is_empty() && self.reachable_frontier.is_empty())
            || self.is_starts_bad();
    }
}

type NodeId = usize;

pub fn union_pdr(pcp: PCP) -> bool {
    let mut g: Graph = Graph::new(pcp.clone());
    println!(
        "{:?}",
        g.starts
            .iter()
            .map(|s| g.nodes[*s].seq.clone())
            .collect_vec()
    );

    while !g.step() {}
    g.print_graph_dot();

    if g.is_starts_bad() {
        println!("bad");
        for s in g.starts.iter() {
            if g.nodes[*s].bad {
                println!("{:?}", g.get_bad_path(*s));
            }
        }
        return false;
    } else {
        println!("closed");
        println!("{:?}", g.get_invariant());
        return true;
    }
}
