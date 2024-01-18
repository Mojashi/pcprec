mod pcpseq;
mod pcp;

use itertools::Itertools;
use pcp::{PCP, parse_pcp_string, Tile};
use pcpseq::{PCPSequence, MidExactSequence};
use regex::Regex;

use std::{
    cmp::min,
    collections::{HashSet, VecDeque},
    io::{BufRead, Write},
};

use crate::pcpseq::{ExactSequence, PCPDir, MidWildSequence};

fn substrings(s: &str, max_len: usize) -> Vec<String> {
    let mut ret: Vec<String> = (0..min(s.len(), max_len + 1))
        .flat_map(|l| -> Vec<String> {
            (0..s.len() - l).map(|i| s[i..i + l].to_string()).collect()
        })
        .collect();
    ret.sort();
    ret.dedup();
    ret
}

fn abstract_seq(seq: &PCPSequence) -> Vec<PCPSequence> {
    let max_len = 20;
    match seq {
        PCPSequence::Exact(e) => {
            let mut ret = substrings(&e.seq, max_len)
                .into_iter()
                .map(|s| PCPSequence::MidExact(MidExactSequence { mid: s, dir: e.dir }))
                .collect_vec();

            ret.append(&mut (1..e.seq.len()).flat_map(|st| 
                (st..e.seq.len()).map(|ens| PCPSequence::MidWild(MidWildSequence { 
                    front: e.seq[..st].to_string(), 
                    back: e.seq[ens..].to_string(),
                    dir: e.dir,
                })).collect_vec()
            ).collect_vec());

            ret.push(seq.clone());
            ret
        }
        PCPSequence::MidWild(e) => {
            vec![PCPSequence::MidWild(MidWildSequence {
                front: e.front[..min(e.front.len(), max_len)].to_string(),
                back: e.back[(e.back.len() - min(e.back.len(), max_len))..].to_string(),
                dir: e.dir,
            })]
        }
        PCPSequence::MidExact(e) => {
            let mut ret = substrings(&e.mid, max_len)
                .into_iter()
                .map(|s| PCPSequence::MidExact(MidExactSequence { mid: s, dir: e.dir }))
                .collect_vec();
            ret.push(seq.clone());
            ret
        }
    }
}

fn check_recursive(pcp: &PCP, cur: &PCPSequence, assumptions: &mut Vec<PCPSequence>, depth: u32) -> bool {
    //println!("{depth} cur: {:?}", cur);
    if depth > 20 {
        return false;
    }
    //println!("cur: {:?}, assumptions: {:?}", cur, assumptions);
    //println!("{:?}", assumptions.len());
    if cur.contains_empty() {
        log::debug!("contains empty!");
        return false;
    }
    if assumptions.iter().any(|s| -> bool { s.contains(cur) }) {
        log::debug!("assumption hit!");
        return true;
    }
    for s in abstract_seq(cur)
        .into_iter()
        .filter(|s| -> bool { !s.contains_empty() })
    {
        log::debug!("abstracted: {:?}", s);
        let sofar_count = assumptions.len();

        let mut nexts = s.apply_pcp(pcp);
        //nexts.shuffle(&mut thread_rng());
        assumptions.push(s);
        log::debug!("nexts: {:?}", nexts);
        let res = nexts
            .iter()
            .all(|nex| check_recursive(pcp, nex, assumptions, depth + 1));
        if res {
            return true;
        }

        assert!(assumptions.len() >= sofar_count);
        assumptions.truncate(sofar_count);
    }
    false
}

#[derive(Debug, Clone)]
struct Result {
    assumptions: Vec<PCPSequence>,
    result: bool,
    start: PCPSequence,
}

fn pdr_like(pcp: &PCP) -> (bool, Vec<Result>) {
    log::debug!("pcp: {:?}", pcp);
    let firsts = ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    }
    .apply_pcp(&pcp);
    log::debug!("firsts: {:?}", firsts);

    let results = firsts
        .iter()
        .map(|s| -> Result {
            let mut assumptions = vec![];
            let r = check_recursive(&pcp, &PCPSequence::Exact(s.clone()), &mut assumptions, 0);
            Result {
                assumptions: assumptions,
                result: r,
                start: PCPSequence::Exact(s.clone()),
            }
        })
        .collect_vec();

    let res = results.iter().all(|s| -> bool { s.result });
    log::debug!("final result: {res} {:?}", pcp);
    (res, results)
}

fn enumerate_substrings_from_pcp(pcp: &PCP, max_len: usize) -> (Vec<String>, Vec<String>) {
    let mut q = VecDeque::new();
    q.push_back(ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    });
    let mut visited = vec![];

    for _ in 0..1000 {
        let seq = q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            visited.push(n.clone());
            q.push_back(n);
        }
    }

    visited.sort();
    visited.dedup();

    let mut upperSubStrings = vec![];
    let mut lowerSubStrings = vec![];

    for s in visited {
        if s.dir == PCPDir::UP {
            upperSubStrings.append(&mut substrings(&s.seq, max_len));
        } else {
            lowerSubStrings.append(&mut substrings(&s.seq, max_len));
        }
    }

    upperSubStrings.sort();
    upperSubStrings.dedup();
    lowerSubStrings.sort();
    lowerSubStrings.dedup();

    (upperSubStrings, lowerSubStrings)
}

fn parse_file(f: &str) -> Vec<(String, PCP)> {
    let mut ret = vec![];
    let f = std::fs::File::open(f).unwrap();
    let lines = std::io::BufReader::new(f).lines().collect_vec();
    for result in lines.iter() {
        let record = result.as_ref().unwrap();
        let pcp = parse_pcp_string(&record.as_str());
        ret.push((record.to_string(), pcp));
    }
    ret
}

fn find_rec_substrs(pcp: &PCP) -> (Vec<String>, Vec<String>) {
    let substr_len = 20;
    let (upper_substrs, lower_substrs) = enumerate_substrings_from_pcp(&pcp, substr_len);
    let mut upper_trues: Vec<String> = vec![];

    for s in upper_substrs
        .into_iter()
        .filter(|s| -> bool { s.len() < substr_len })
        .sorted_by_key(|s| s.len())
    {
        if upper_trues.iter().any(|t| s.contains(t)) {
            continue;
        }
        let result = check_recursive(
            &pcp,
            &PCPSequence::Exact(ExactSequence {
                seq: s.clone(),
                dir: PCPDir::UP,
            }),
            &mut vec![], 0
        );
        if result {
            upper_trues.push(s);
        }
    }

    let mut lower_trues: Vec<String> = vec![];
    for s in lower_substrs
        .into_iter()
        .filter(|s| -> bool { s.len() < substr_len })
        .sorted_by_key(|s| s.len())
    {
        if lower_trues.iter().any(|t| s.contains(t)) {
            continue;
        }
        let result = check_recursive(
            &pcp,
            &PCPSequence::Exact(ExactSequence {
                seq: s.clone(),
                dir: PCPDir::DN,
            }),
            &mut vec![], 0
        );
        if result {
            lower_trues.push(s);
        }
    }

    (upper_trues, lower_trues)
}

fn get_recs_for_pcps() {
    for (idx, (raw, pcp)) in parse_file("a.csv").iter().enumerate() {
        println!("pcp: {:?} {:?}", idx, pcp);
        let (upper_trues, lower_trues) = find_rec_substrs(&pcp);
        std::fs::write(
            "results/".to_string() + &idx.to_string() + ".txt",
            format!("{:?}\n{:?}\n{:?}\n", raw, upper_trues, lower_trues),
        )
        .unwrap();

        let (upper_trues, lower_trues) = find_rec_substrs(&pcp.reverse_pcp());
        std::fs::write(
            "results/".to_string() + &idx.to_string() + "-rev.txt",
            format!("{:?}\n{:?}\n{:?}\n", raw, upper_trues, lower_trues),
        )
        .unwrap();
    }
}


fn from_input() {
    let mut input_str = String::new();
    std::io::stdin().read_line(&mut input_str).unwrap();
    let pcp = parse_pcp_string(&input_str);
    println!("pcp: {:?}", pcp);

    let (upper_trues, lower_trues) = find_rec_substrs(&pcp);
    println!("upper recursive strings: {:?}", upper_trues);
    println!("lower recursive strings: {:?}", lower_trues);
}

fn find_answer(pcp: &PCP) -> bool {
    let mut q: VecDeque<ExactSequence> = VecDeque::new();
    let s = ExactSequence {
        seq: "".to_string(),
        dir: PCPDir::UP,
    };
    q.push_back(s);
    let mut visited: HashSet<ExactSequence> = HashSet::new();

    for _ in 0..10000 {
        if q.len() == 0 {
            break;
        }
        let seq = &q.pop_front().unwrap();
        let next = seq.apply_pcp(pcp);
        for n in next {
            if visited.contains(&n) {
                continue;
            }
            if n.seq.len() == 0 {
                return true;
            }

            visited.insert(n.clone());
            q.push_back(n);
        }
    }
    false
}
/*
Format of an instance:
   1st line: index
   2nd line: size, width, optimal length, number of optimal solutions
   3rd & 4th lines: pairs
   5th line: a line break

Instance 1:
3 3 75 2
110   1     0     
1     0     110     
*/
fn parse_instance_list(fname: &str) -> Vec<PCP> {
    let mut ret = vec![];
    let f = std::fs::File::open(fname).unwrap();
    let mut content = std::io::BufReader::new(f).lines().collect_vec().iter().map(|s| s.as_ref().unwrap().to_string()).collect_vec();
    let seperator = Regex::new(r"\s+").expect("Invalid regex");

    while content.is_empty() == false {
        let startline = content
            .iter()
            .enumerate()
            .find(|(_, s)| s.starts_with("Instance"))
            .unwrap()
            .0;
        let ups = seperator.split(&content[startline + 2]).collect_vec();
        let dns = seperator.split(&content[startline + 3]).collect_vec();

        ret.push(PCP {
            tiles: ups.iter().zip(dns.iter())
                .map(|(u, d)| Tile { up: u.to_string(), dn: d.to_string() })
                .filter(|t| -> bool {
                    t.up.len() > 0 && t.dn.len() > 0
                }).collect_vec()
        });
        content = content[startline + 5..].to_vec();
    }

    ret
}

fn apply_pdr(rev: bool) {
    for (idx, (raw, pcp)) in parse_file("a.csv").into_iter().enumerate() {
        let mut pcp = pcp;
        println!("{idx} reversed:{rev} pcp: {:?}", raw);
        if rev {
            pcp = pcp.reverse_pcp();
        }
        let (res, results) = pdr_like(&pcp);
        if res == false {
            continue;
        }
        let results = results.into_iter().map(|r| refine_proof(&pcp, r)).collect_vec();
        println!("result: {:?}", res);

        let mut file =
            std::fs::File::create("proofs/".to_string() + &idx.to_string() + ".txt").unwrap();
        file.write(format!("pcp: {:?}\n", raw).as_bytes()).unwrap();
        file.write(format!("reversed: {:?}\n", rev).as_bytes()).unwrap();
        file.write(format!("result: {:?}\n", res).as_bytes())
            .unwrap();
        for r in results {
            file.write(format!("{:?}\n", r).as_bytes()).unwrap();
        }
    }
}

// fn check_recursive_bounded(pcp: &PCP, s: &str, dir: PCPDir) -> bool {
//     let mut q = VecDeque::new();

//     q.push_back(PCPSequence::MidWild(MidWildSequence {
//         front: s.to_string(),
//         back: "".to_string(),
//         dir: dir,
//     }));

//     //println!("q: {:?}", q);

//     let mut visited: HashSet<PCPSequence> = HashSet::new();

//     let mut dot_string = String::new();
//     dot_string += "digraph {\n";
//     let mut reced = false;

//     while visited.len() < 1000 && q.len() > 0 {
//         let seq = q.pop_front().unwrap();

//         if visited.contains(&seq) {
//             continue;
//         }
//         visited.insert(seq.clone());

//         let next = seq.apply_pcp(pcp);
//         //println!("{:?} -> {:?}", seq, next);
//         dot_string +=
//             ("\"".to_owned() + &format!("{:?}", seq).replace("\"", "") + "\" -> {").as_str();
//         for n in &next {
//             dot_string += ("\"".to_owned() + &format!("{:?}", n).replace("\"", "") + "\"").as_str();
//         }
//         dot_string += "}\n";

//         for n in next {
//             if n.dir() == dir && n.contains_str(s) {
//                 reced = true;
//                 continue;
//             }
//             if visited.contains(&n) {
//                 continue;
//             }
//             if match &n {
//                 PCPSequence::Exact(e) => e.seq.len() == 0,
//                 _ => false,
//             } {
//                 visited.insert(n.clone());
//                 break;
//             };
//             q.push_back(n);
//         }
//     }

//     dot_string += "}\n";
//     //std::fs::write("graph.dot", dot_string).unwrap();

//     let unreached_empty = visited.iter().all(|s| match s {
//         PCPSequence::Exact(e) => e.seq.len() != 0,
//         _ => true,
//     });

//     //println!("unreached_empty: {unreached_empty}");
//     q.len() == 0 && unreached_empty && reced
// }

fn check_valid_proof(pcp: &PCP, proof: &Vec<PCPSequence>) -> bool {
    if proof.iter().any(|p| p.contains_empty()) {
        return false;
    }
    for p in proof {
        let nexts = p.apply_pcp(pcp);
        for n in nexts {
            if proof.iter().any(|p: &PCPSequence| p.contains(&n)) {
                continue;
            }
            return false;
        }
    }
    true
}

fn refine_proof(pcp: &PCP, result: Result) -> Result {
    let mut ret: Vec<PCPSequence> = result.assumptions.clone();

    if !check_valid_proof(pcp, &result.assumptions) {
        panic!("invalid proof");
    }

    for (idx, a) in result.assumptions.iter().enumerate().rev() {
        ret.remove(idx);
        if !check_valid_proof(pcp, &ret) {
            ret.insert(idx, a.clone());
        }
    }
    Result {
        assumptions: ret,
        ..result
    }
}

fn hard_instances_check() {
    let pcps = parse_instance_list("200hard.txt");
    for pcp in pcps.iter() {
        println!("pcp: {:?}", pcp);
        let (res, results) = pdr_like(pcp);
        println!("result: {:?}", res);
        if res {
            let refined = refine_proof(&pcp, results[0].clone());
            println!("refined: {:?}", refined);
            println!("{:?} -> {:?}", results[0].assumptions.len(), refined.assumptions.len());
            panic!("fail");
        }
    }
}

fn main() {
    //hard_instances_check();
    apply_pdr(false);
    apply_pdr(true);
    // for _ in 0..1000 {
    //     let pcp = gen_random_pcp(3, 4);
    //     println!("{:?}", pcp);

    //     if find_answer(&pcp) {
    //         println!("found!");

    //         let res = pdr_like(pcp);
    //         if res {
    //             panic!("fail");
    //         }
    //     }
    // }
    //get_recs_for_pcps();
    //from_input();
    //explor_main(true);

    //pdr_like(parse_pcp_string("PCP(Vector(Tile(1111,110), Tile(1011,1), Tile(1,1111)))"));
    // let pcp = parse_pcp_string("PCP(Vector(Tile(1111,110), Tile(1110,1), Tile(1,1111)))");
    // pdr_like(swap_pcp(&reverse_pcp(&pcp)));

    // let pcp = parse_pcp_string("PCP(Vector(Tile(1110,01), Tile(1101,11), Tile(1,1011)))");
    // let result = check_recursive(&pcp, "111011110", PCPDir::UP);
    // println!("result: {:?}", result);
}
