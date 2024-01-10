mod automaton;

use itertools::Itertools;
use regex::Regex;

#[derive(Debug)]
struct Tile {
    up: String,
    dn: String,
}

#[derive(Debug)]
struct PCP {
    tiles: Vec<Tile>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
enum PCPDir {
    UP,
    DN,
}

impl PCPDir {
    fn opposite(&self) -> PCPDir {
        match self {
            PCPDir::UP => PCPDir::DN,
            PCPDir::DN => PCPDir::UP,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
struct MidWildSequence {
    front_wild: bool,
    front: String,
    back: String,
    dir: PCPDir,
}

impl MidWildSequence {
    fn contains_str(&self, s: &str) -> bool {
        self.front.contains(s) || self.back.contains(s)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
struct ExactSequence {
    seq: String,
    dir: PCPDir,
}

impl ExactSequence {
    fn contains_str(&self, s: &str) -> bool {
        self.seq.contains(s)
    }
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
enum PCPSequence {
    MidWild(MidWildSequence),
    Exact(ExactSequence),
}

impl PCPSequence {
    fn contains_str(&self, s: &str) -> bool {
        match self {
            PCPSequence::MidWild(seq) => seq.contains_str(s),
            PCPSequence::Exact(seq) => seq.contains_str(s),
        }
    }

    fn apply_pcp(&self, pcp: &PCP) -> Vec<PCPSequence> {
        pcp.tiles
            .iter()
            .flat_map(|tile| self.apply_tile(tile))
            .collect_vec()
    }

    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        match self {
            PCPSequence::MidWild(seq) => seq.apply_tile(tile),
            PCPSequence::Exact(seq) => seq.apply_tile(tile),
        }
    }

    fn dir(&self) -> PCPDir {
        match self {
            PCPSequence::MidWild(seq) => seq.dir,
            PCPSequence::Exact(seq) => seq.dir,
        }
    }

    fn swap_dir(&self) -> PCPSequence {
        match self {
            PCPSequence::MidWild(seq) => PCPSequence::MidWild(MidWildSequence {
                front_wild: seq.front_wild,
                front: seq.front.clone(),
                back: seq.back.clone(),
                dir: seq.dir.opposite(),
            }),
            PCPSequence::Exact(seq) => PCPSequence::Exact(ExactSequence {
                seq: seq.seq.clone(),
                dir: seq.dir.opposite(),
            }),
        }
    }
}

impl ExactSequence {
    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        if self.dir == PCPDir::DN {
            return ExactSequence {
                seq: self.seq.clone(),
                dir: self.dir.opposite(),
            }
            .apply_tile(&swap_tile(tile))
            .iter()
            .map(|s| s.swap_dir())
            .collect_vec();
        }

        let upper = self.seq.clone() + &tile.up;
        if upper.starts_with(&tile.dn) {
            return vec![PCPSequence::Exact(ExactSequence {
                seq: upper[tile.dn.len()..].to_string(),
                dir: self.dir,
            })];
        }

        if tile.dn.starts_with(&upper) {
            return vec![PCPSequence::Exact(ExactSequence {
                seq: tile.dn[upper.len()..].to_string(),
                dir: self.dir.opposite(),
            })];
        }

        return vec![];
    }
}

impl MidWildSequence {
    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        if self.dir == PCPDir::DN {
            return MidWildSequence {
                front_wild: self.front_wild,
                front: self.front.clone(),
                back: self.back.clone(),
                dir: self.dir.opposite(),
            }
            .apply_tile(&swap_tile(tile))
            .iter()
            .map(|s| s.swap_dir())
            .collect_vec();
        }
        if self.front.len() == 0 {
            // * -> *
            if self.back.len() == 0 {
                return vec![PCPSequence::MidWild(self.clone())];
            }
            return vec![PCPSequence::MidWild(MidWildSequence {
                front_wild: true,
                front: self.back.clone(),
                back: "".to_string(),
                dir: self.dir,
            })];
        }

        let mut ret: Vec<PCPSequence> = vec![];

        let dn_start_max = if self.front_wild {
            tile.dn.len() as i32 - 1
        } else {
            0
        };
        for dn_start in 0..=dn_start_max {
            let rest_dn = &tile.dn[dn_start as usize..];

            if self.front.starts_with(rest_dn) {
                ret.push(PCPSequence::MidWild(MidWildSequence {
                    front_wild: false,
                    front: self.front[rest_dn.len()..].to_string(),
                    back: self.back.clone() + &tile.up,
                    dir: self.dir,
                }));
            }

            if rest_dn.starts_with(&self.front) {
                ret.push(PCPSequence::MidWild(MidWildSequence {
                    front_wild: false,
                    front: "".to_string(),
                    back: self.back.clone() + &tile.up,
                    dir: self.dir,
                }));

                for mid_chars in 0..rest_dn.len() - self.front.len() {
                    let suf = rest_dn[self.front.len() + mid_chars..].to_string();
                    let upper = self.back.clone() + &tile.up;
                    if upper.starts_with(&suf) {
                        ret.push(PCPSequence::Exact(ExactSequence {
                            seq: upper[suf.len()..].to_string(),
                            dir: self.dir,
                        }));
                    }

                    if suf.starts_with(&upper) {
                        ret.push(PCPSequence::Exact(ExactSequence {
                            seq: suf[upper.len()..].to_string(),
                            dir: self.dir.opposite(),
                        }))
                    }
                }
            }
        }

        ret.dedup();
        ret
    }
}

// parse string like PCP(Vector(Tile(1110,1), Tile(1,0), Tile(0,1110)))
fn parse_pcp_string(s: &str) -> PCP {
    let r = Regex::new(r"Tile\((\d+),(\d+)\)").unwrap();

    let tiles = r
        .captures_iter(s)
        .map(|cap| Tile {
            up: cap[1].to_string(),
            dn: cap[2].to_string(),
        })
        .collect();

    PCP { tiles: tiles }
}

fn swap_tile(tile: &Tile) -> Tile {
    Tile {
        up: tile.dn.clone(),
        dn: tile.up.clone(),
    }
}

fn swap_pcp(pcp: &PCP) -> PCP {
    PCP {
        tiles: pcp.tiles.iter().map(|tile| swap_tile(tile)).collect(),
    }
}

use std::{collections::{VecDeque, HashSet}, path::Iter, io::BufRead};

fn check_recursive(pcp: &PCP, s: &str, dir: PCPDir) -> bool {
    let mut q = VecDeque::new();

    q.push_back(PCPSequence::MidWild(MidWildSequence {
        front_wild: true,
        front: s.to_string(),
        back: "".to_string(),
        dir: dir,
    }));

    //println!("q: {:?}", q);

    let mut visited: HashSet<PCPSequence> = HashSet::new();

    let mut dot_string = String::new();
    dot_string += "digraph {\n";
    let mut reced = false;

    while visited.len() < 1000 && q.len() > 0 {
        let seq = q.pop_front().unwrap();

        if visited.contains(&seq) {
            continue;
        }
        visited.insert(seq.clone());

        let next = seq.apply_pcp(pcp);
        //println!("{:?} -> {:?}", seq, next);
        dot_string +=
            ("\"".to_owned() + &format!("{:?}", seq).replace("\"", "") + "\" -> {").as_str();
        for n in &next {
            dot_string += ("\"".to_owned() + &format!("{:?}", n).replace("\"", "") + "\"").as_str();
        }
        dot_string += "}\n";

        for n in next {
            if n.dir() == dir && n.contains_str(s) {
                reced = true;
                continue;
            }
            if visited.contains(&n) {
                continue;
            }
            if match &n {
                PCPSequence::Exact(e) => e.seq.len() == 0,
                _ => false,
            } {
                visited.insert(n.clone());
                break;
            };
            q.push_back(n);
        }
    }

    dot_string += "}\n";
    //std::fs::write("graph.dot", dot_string).unwrap();

    let unreached_empty = visited.iter().all(|s| match s {
        PCPSequence::Exact(e) => e.seq.len() != 0,
        _ => true,
    });

    //println!("unreached_empty: {unreached_empty}");
    q.len() == 0 && unreached_empty && reced
}

fn enumerate_01_strings(max_len: usize) -> Vec<String> {
    let mut ret = vec![];
    for len in 0..=max_len {
        for i in 0..2usize.pow(len as u32) {
            let mut s = format!("{:b}", i);
            while s.len() < len {
                s = "0".to_string() + &s;
            }
            ret.push(s);
        }
    }
    ret
}

fn substrings(s: &str) -> Vec<String> {
    let mut ret = vec![];
    for i in 0..s.len() {
        for j in i+1..s.len() {
            ret.push(s[i..j].to_string());
        }
    }
    ret
}

fn explore_pcp(pcp: &PCP) -> (Vec<String>, Vec<String>) {
    let mut q = VecDeque::new();
    q.push_back(PCPSequence::Exact(ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    }));
    let mut visited = vec![];

    for i in 0..1000 {
        let seq = q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            visited.push(n.clone());
            q.push_back(n);
        }
    };

    visited.sort();
    visited.dedup();

    let mut upperSubStrings = vec![];
    let mut lowerSubStrings = vec![];

    for s in visited {
        match s {
            PCPSequence::Exact(e) => {
                if e.dir == PCPDir::UP {
                    upperSubStrings.append(&mut substrings(&e.seq));
                } else {
                    lowerSubStrings.append(&mut substrings(&e.seq));
                }
            },
            _ => {},
        }
    };

    upperSubStrings.sort();
    upperSubStrings.dedup();
    lowerSubStrings.sort();
    lowerSubStrings.dedup();

    (upperSubStrings, lowerSubStrings)
}

fn parse_file(f: &str) -> Vec<PCP> {
    let mut ret = vec![];
    let f = std::fs::File::open(f).unwrap();
    let lines = std::io::BufReader::new(f).lines().collect_vec();
    for result in lines.iter() {
        let record = result.as_ref().unwrap();
        let pcp = parse_pcp_string(&record.as_str());
        ret.push(pcp);
    }
    ret
}

fn get_recs_for_pcps() {
    for (idx,pcp) in parse_file("a.csv").iter().enumerate() {
        println!("pcp: {:?}", pcp);
        let (upperSubStrings, lowerSubStrings) = explore_pcp(&pcp);
        let mut upper_trues: Vec<String> = vec![];
        for s in upperSubStrings.iter().filter(|s| -> bool {s.len() < 10}) {
            let result = check_recursive(&pcp, &s, PCPDir::UP);
            if result {
                upper_trues.push(s.clone());
            }
        }
        println!("upper recursive strings: {:?}", upper_trues);

        let mut lower_trues: Vec<String> = vec![];
        for s in lowerSubStrings.iter().filter(|s| -> bool {s.len() < 10}) {
            let result = check_recursive(&pcp, &s, PCPDir::DN);
            if result {
                lower_trues.push(s.clone());
            }
        }
        println!("lower recursive strings: {:?}", lower_trues);

        std::fs::write("results/".to_string() + &idx.to_string() + ".txt", format!("{:?}\n{:?}\n{:?}\n",pcp, upper_trues, lower_trues)).unwrap();
    }
}

fn from_input() {
    let mut input_str = String::new();
    std::io::stdin().read_line(&mut input_str).unwrap();
    let pcp = parse_pcp_string(&input_str);
    println!("pcp: {:?}", pcp);


    let (upperSubStrings, lowerSubStrings) = explore_pcp(&pcp);
    
    let mut upper_trues: Vec<String> = vec![];
    for s in upperSubStrings.iter().filter(|s| -> bool {s.len() < 10}) {
        let result = check_recursive(&pcp, &s, PCPDir::UP);
        if result {
            upper_trues.push(s.clone());
        }
    }
    println!("upper recursive strings: {:?}", upper_trues);

    let mut lower_trues: Vec<String> = vec![];
    for s in lowerSubStrings.iter().filter(|s| -> bool {s.len() < 10}) {
        let result = check_recursive(&pcp, &s, PCPDir::DN);
        if result {
            lower_trues.push(s.clone());
        }
    }
    println!("lower recursive strings: {:?}", lower_trues);
}
fn main() {
    get_recs_for_pcps();
}
