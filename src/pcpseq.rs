use crate::pcp::{Tile, PCP};
use itertools::Itertools;

#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub enum PCPDir {
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

fn enumerate01strings(len: usize) -> Vec<String> {
    if len == 0 {
        return vec!["".to_string()];
    }
    let mut ret = vec![];
    for s in enumerate01strings(len - 1) {
        ret.push(s.clone() + "0");
        ret.push(s + "1");
    }
    ret
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
pub struct MidWildSequence {
    pub front: String,
    pub back: String,
    pub dir: PCPDir,
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
pub struct MidExactSequence {
    pub mid: String,
    pub dir: PCPDir,
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
pub struct ExactSequence {
    pub seq: String,
    pub dir: PCPDir,
}

impl ExactSequence {
    fn contains_empty(&self) -> bool {
        self.seq.len() == 0
    }
    fn contains(&self, s: &PCPSequence) -> bool {
        match s {
            PCPSequence::Exact(e) => {
                self.dir == s.dir() && self.seq == e.seq || self.contains_empty() && e.seq == ""
            }
            PCPSequence::MidWild(_) => false,
            PCPSequence::MidExact(_) => false,
        }
    }
}

impl MidExactSequence {
    fn sample(&self) -> Vec<String> {
        let front = (0..5)
            .into_iter()
            .flat_map(|midlen| enumerate01strings(midlen));
        let back = (0..5)
            .into_iter()
            .flat_map(|midlen| enumerate01strings(midlen));
        front
            .cartesian_product(back)
            .map(|(f, b)| vec![f, self.mid.clone(), b].concat())
            .collect_vec()
    }

    fn contains_empty(&self) -> bool {
        self.mid.len() == 0
    }
    fn contains(&self, s: &PCPSequence) -> bool {
        match s {
            PCPSequence::Exact(e) => {
                self.dir == s.dir() && e.seq.contains(&self.mid)
                    || self.contains_empty() && e.seq == ""
            }
            PCPSequence::MidExact(e) => self.dir == s.dir() && e.mid.contains(&self.mid),
            PCPSequence::MidWild(e) => {
                self.dir == s.dir() && (e.front.contains(&self.mid) || e.back.contains(&self.mid))
            }
        }
    }
}

#[test]
fn contain_test() {
    let a = PCPSequence::MidExact(MidExactSequence { mid: "0110".to_owned(), dir: PCPDir::DN });
    let b = PCPSequence::MidWild(MidWildSequence { front: "".to_owned(), back: "11101101".to_owned(), dir: PCPDir::DN });
    assert!(a.contains(&b));
}

#[test]
fn exact_seq_test() {
    let s = PCPSequence::Exact(ExactSequence{
        seq: "110".to_string(),
        dir: PCPDir::DN,
    });
    let tile = Tile { up: "1101".to_string(), dn: "110".to_string() };
    let nexts = s.apply_tile(&tile);
    
    assert_eq!(nexts.len(), 1);
    assert_eq!(nexts[0], PCPSequence::Exact(ExactSequence{
        seq: "10".to_string(),
        dir: PCPDir::DN,
    }));
}

impl MidWildSequence {
    fn sample(&self) -> Vec<String> {
        (0..5)
            .into_iter()
            .flat_map(|midlen| {
                enumerate01strings(midlen)
                    .into_iter()
                    .map(|mid| vec![self.front.clone(), mid, self.back.clone()].concat())
            })
            .collect_vec()
    }

    fn contains_empty(&self) -> bool {
        self.front.len() == 0 && self.back.len() == 0
    }
    fn contains(&self, s: &PCPSequence) -> bool {
        match s {
            PCPSequence::Exact(e) => {
                self.dir == s.dir()
                    && e.seq.len() >= self.front.len() + self.back.len()
                    && e.seq.starts_with(&self.front)
                    && e.seq.ends_with(&self.back)
                    || self.contains_empty() && e.seq == ""
            }
            PCPSequence::MidWild(e) => {
                self.dir == s.dir()
                    && e.front.starts_with(&self.front)
                    && e.back.ends_with(&self.back)
            }
            PCPSequence::MidExact(_) => false,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, PartialOrd, Ord, Hash)]
pub enum PCPSequence {
    MidWild(MidWildSequence),
    Exact(ExactSequence),
    MidExact(MidExactSequence),
}

impl PCPSequence {
    pub fn num_chars(&self) -> u32 {
        match self {
            PCPSequence::Exact(e) => e.seq.len() as u32,
            PCPSequence::MidWild(e) => e.front.len() as u32 + e.back.len() as u32,
            PCPSequence::MidExact(e) => e.mid.len() as u32,
        }
    }

    pub fn sample(&self) -> Vec<String> {
        match self {
            PCPSequence::Exact(e) => vec![e.seq.clone()],
            PCPSequence::MidWild(e) => e.sample(),
            PCPSequence::MidExact(e) => e.sample(),
        }
    }

    pub fn contains_empty(&self) -> bool {
        match self {
            PCPSequence::Exact(e) => e.contains_empty(),
            PCPSequence::MidWild(e) => e.contains_empty(),
            PCPSequence::MidExact(e) => e.contains_empty(),
        }
    }
    pub fn contains(&self, s: &PCPSequence) -> bool {
        let ret = match self {
            PCPSequence::Exact(e) => e.contains(s),
            PCPSequence::MidWild(e) => e.contains(s),
            PCPSequence::MidExact(e) => e.contains(s),
        };

        // match s {
        //     PCPSequence::Exact(_) => {}
        //     _ =>
        //     if ret  {
        //         s.sample().iter().for_each(|ss| {
        //             if !self.contains(&PCPSequence::Exact(ExactSequence { seq: ss.clone(), dir: self.dir() })) {
        //                 panic!("invalid self: {:?} s: {:?}  ss: {:?}", self, s, ss);
        //             }
        //         })
        //     }
        // }

        ret
    }

    pub fn apply_pcp(&self, pcp: &PCP) -> Vec<PCPSequence> {
        let ret = pcp.tiles
            .iter()
            .flat_map(|tile| self.apply_tile(tile))
            .sorted()
            .dedup()
            .collect_vec();

        // ret.into_iter()
        //     .flat_map(|f| match f {
        //         PCPSequence::Exact(e) => vec![PCPSequence::Exact(e)],
        //         PCPSequence::MidWild(e) => {
        //             if e.front.len() > 0 {
        //                 PCPSequence::MidWild(e.clone()).apply_pcp(pcp)
        //             }
        //             else {
        //                 vec![PCPSequence::MidExact(MidExactSequence { mid: e.back.clone(), dir: e.dir})]
        //             }
        //         }
        //         PCPSequence::MidExact(e) => vec![PCPSequence::MidExact(e)],
        //     })
        //     .collect_vec()
        ret
    }

    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        let ret = match self {
            PCPSequence::MidWild(seq) => seq.apply_tile(tile),
            PCPSequence::Exact(seq) => seq
                .apply_tile(tile)
                .into_iter()
                .map(|f| PCPSequence::Exact(f))
                .collect_vec(),
            PCPSequence::MidExact(seq) => seq.apply_tile(tile),
        };

        // match self {
        //     PCPSequence::Exact(_) => {}
        //     _ => {
        //         let valid = self
        //             .sample()
        //             .into_iter()
        //             .flat_map(|s| {
        //                 PCPSequence::Exact(ExactSequence {
        //                     seq: s.clone(),
        //                     dir: self.dir(),
        //                 })
        //                 .apply_tile(tile)
        //                 .into_iter()
        //                 .map(|n| (s.clone(), n)).collect_vec()
        //             })
        //             .find(|(o, s)| ret.iter().all(|f| !f.contains(&s)));
        //         match valid {
        //             None => {}
        //             Some(cex) => {
        //                 panic!(
        //                     "invalid tile {:?} for {:?}. ret: {:?} cex: {:?}",
        //                     tile, self, ret, cex
        //                 );
        //             }
        //         }
        //     }
        // }

        ret
    }

    pub fn dir(&self) -> PCPDir {
        match self {
            PCPSequence::MidWild(seq) => seq.dir,
            PCPSequence::Exact(seq) => seq.dir,
            PCPSequence::MidExact(seq) => seq.dir,
        }
    }

    pub fn swap_dir(&self) -> PCPSequence {
        match self {
            PCPSequence::MidWild(seq) => PCPSequence::MidWild(MidWildSequence {
                front: seq.front.clone(),
                back: seq.back.clone(),
                dir: seq.dir.opposite(),
            }),
            PCPSequence::Exact(seq) => PCPSequence::Exact(ExactSequence {
                seq: seq.seq.clone(),
                dir: seq.dir.opposite(),
            }),
            PCPSequence::MidExact(seq) => PCPSequence::MidExact(MidExactSequence {
                mid: seq.mid.clone(),
                dir: seq.dir.opposite(),
            }),
        }
    }
}

impl ExactSequence {
    pub fn apply_pcp(&self, pcp: &PCP) -> Vec<ExactSequence> {
        pcp.tiles
            .iter()
            .flat_map(|tile| self.apply_tile(tile))
            .collect_vec()
    }
    pub fn swap_dir(&self) -> ExactSequence {
        ExactSequence {
            seq: self.seq.clone(),
            dir: self.dir.opposite(),
        }
    }
    fn apply_tile(&self, tile: &Tile) -> Vec<ExactSequence> {
        if self.dir == PCPDir::DN {
            return ExactSequence {
                seq: self.seq.clone(),
                dir: self.dir.opposite(),
            }
            .apply_tile(&tile.swap_tile())
            .into_iter()
            .map(|s| s.swap_dir())
            .collect_vec();
        }

        let upper = self.seq.clone() + &tile.up;
        if upper.starts_with(&tile.dn) {
            return vec![ExactSequence {
                seq: upper[tile.dn.len()..].to_string(),
                dir: self.dir,
            }];
        }

        if tile.dn.starts_with(&upper) {
            return vec![ExactSequence {
                seq: tile.dn[upper.len()..].to_string(),
                dir: self.dir.opposite(),
            }];
        }

        return vec![];
    }
}


#[test]
fn midwild_apply_test() {
    let s = PCPSequence::MidWild(MidWildSequence { front: "11".to_string(), back: "011011".to_string(), dir: PCPDir::UP });
}

impl MidWildSequence {
    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        if self.dir == PCPDir::DN {
            return MidWildSequence {
                front: self.front.clone(),
                back: self.back.clone(),
                dir: self.dir.opposite(),
            }
            .apply_tile(&tile.swap_tile())
            .into_iter()
            .map(|s| s.swap_dir())
            .collect_vec();
        }
        if self.front.len() == 0 {
            // * -> *
            if self.back.len() == 0 {
                return vec![PCPSequence::MidExact(MidExactSequence{mid: "".to_string(), dir:PCPDir::UP})];
            }
        }

        let mut ret: Vec<PCPSequence> = vec![];
        if self.front.len() == 0 {
            ret.extend(PCPSequence::MidExact(MidExactSequence {
                mid: self.back.clone(),
                dir: self.dir,
            }).apply_tile(tile));
        }

        if self.front.starts_with(&tile.dn) {
            ret.push(PCPSequence::MidWild(MidWildSequence {
                front: self.front[tile.dn.len()..].to_string(),
                back: self.back.clone() + &tile.up,
                dir: self.dir,
            }));
        }

        if self.front.len() > 0 && tile.dn.starts_with(&self.front) {
            ret.push(PCPSequence::MidWild(MidWildSequence {
                front: "".to_string(),
                back: self.back.clone() + &tile.up,
                dir: self.dir,
            }));

            for mid_chars in 0..tile.dn.len() - self.front.len() {
                let suf = tile.dn[self.front.len() + mid_chars..].to_string();
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
        ret.dedup();
        ret
    }
}


impl MidExactSequence {
    fn apply_tile(&self, tile: &Tile) -> Vec<PCPSequence> {
        if self.dir == PCPDir::DN {
            return MidExactSequence {
                mid: self.mid.clone(),
                dir: self.dir.opposite(),
            }
            .apply_tile(&tile.swap_tile())
            .into_iter()
            .map(|s| s.swap_dir())
            .collect_vec();
        }

        // 完全に .*self.mid.* の最初の .* に飲まれて、 .*self.mid.*tail は .*self.mid.* に含まれるので無視していい
        let mut ret: Vec<PCPSequence> = vec![PCPSequence::MidExact(self.clone())];

        for start_len in 0..tile.dn.len() {
            let rest_dn = &tile.dn[start_len..];

            if rest_dn.starts_with(&self.mid) {
                ret.push(PCPSequence::MidWild(MidWildSequence {
                    front: "".to_string(),
                    back: tile.up.clone(),
                    dir: self.dir,
                }));

                for mid_chars in 0..rest_dn.len() - self.mid.len() {
                    let suf = rest_dn[self.mid.len() + mid_chars..].to_string();
                    let upper = tile.up.clone();
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
            if self.mid.starts_with(rest_dn) {
                ret.push(PCPSequence::MidWild(MidWildSequence {
                    front: self.mid[rest_dn.len()..].to_string(),
                    back: tile.up.clone(),
                    dir: self.dir,
                }))
            }
        }

        ret.dedup();
        ret
    }
}
