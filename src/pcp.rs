use std::{collections::{HashSet, VecDeque}, rc::Rc};

use itertools::Itertools;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Tile {
    pub up: String,
    pub dn: String,
}

impl Tile {
    pub fn swap_tile(&self) -> Tile {
        Tile {
            up: self.dn.clone(),
            dn: self.up.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PCP {
    pub tiles: Vec<Tile>,
}

impl PCP {
    // parse string like PCP(Vector(Tile(1110,1), Tile(1,0), Tile(0,1110)))
    pub fn parse_pcp_string(s: &str) -> PCP {
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

    pub fn co(&self) -> PCP {
        self.swap_pcp().reverse_pcp()
    }

    pub fn swap_pcp(&self) -> PCP {
        PCP {
            tiles: self.tiles.iter().map(|tile| tile.swap_tile()).collect(),
        }
    }

    pub fn reverse_pcp(&self) -> PCP {
        PCP {
            tiles: self
                .tiles
                .iter()
                .map(|tile| Tile {
                    up: tile.up.chars().rev().collect(),
                    dn: tile.dn.chars().rev().collect(),
                })
                .collect(),
        }
    }
}

fn gen_random_pcp(num_tile: usize, tile_size: usize) -> PCP {
    let mut tiles = vec![];
    for _ in 0..num_tile {
        let mut up = String::new();
        let mut dn = String::new();
        for _ in 0..(rand::random::<usize>() % tile_size + 1) {
            up += (&rand::random::<u8>() % 2).to_string().as_str();
        }
        for _ in 0..(rand::random::<usize>() % tile_size + 1) {
            dn += (&rand::random::<u8>() % 2).to_string().as_str();
        }
        tiles.push(Tile { up, dn });
    }
    PCP { tiles }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
pub enum PCPDir {
    UP,
    DN,
}

impl PCPDir {
    pub fn opposite(&self) -> PCPDir {
        match self {
            PCPDir::UP => PCPDir::DN,
            PCPDir::DN => PCPDir::UP,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct PCPConfig {
    pub seq: String,
    pub dir: PCPDir,
}

impl PCPConfig {
    pub fn apply_pcp(&self, pcp: &PCP) -> Vec<PCPConfig> {
        pcp.tiles
            .iter()
            .flat_map(|tile| self.apply_tile(tile))
            .collect_vec()
    }

    pub fn co(&self) -> PCPConfig {
        self.swap_dir().reverse()
    }

    pub fn swap_dir(&self) -> PCPConfig {
        PCPConfig {
            seq: self.seq.clone(),
            dir: self.dir.opposite(),
        }
    }
    pub fn reverse(&self) -> PCPConfig {
        PCPConfig {
            seq: self.seq.chars().rev().collect(),
            dir: self.dir,
        }
    }
    fn apply_tile(&self, tile: &Tile) -> Vec<PCPConfig> {
        if self.dir == PCPDir::DN {
            return self
                .swap_dir()
                .apply_tile(&tile.swap_tile())
                .into_iter()
                .map(|s| s.swap_dir())
                .collect_vec();
        }

        let upper = self.seq.clone() + &tile.up;
        if upper.starts_with(&tile.dn) {
            return vec![PCPConfig {
                seq: upper[tile.dn.len()..].to_string(),
                dir: PCPDir::UP,
            }];
        }

        if tile.dn.starts_with(&upper) {
            return vec![PCPConfig {
                seq: tile.dn[upper.len()..].to_string(),
                dir: PCPDir::DN,
            }];
        }

        return vec![];
    }
}

impl PCP {
    pub fn enumerate_configurations(&self, size: u32) -> Vec<PCPConfig> {
        let mut q: VecDeque<Rc<PCPConfig>> = VecDeque::new();
        let emp_conf = Rc::new(PCPConfig {
            seq: "".to_string(),
            dir: PCPDir::UP,
        });
        let mut visited: HashSet<Rc<PCPConfig>> = HashSet::new();
        visited.insert(emp_conf.clone());
        q.push_back(emp_conf);

        while visited.len() < size as usize {
            if q.len() == 0 {
                break;
            }
            let seq = q.pop_front().unwrap();
            let next = seq.apply_pcp(self);
            let new_next = next
                .into_iter()
                .filter(|n| !visited.contains(n))
                .map(|n| Rc::new(n))
                .collect_vec();
            //println!("{:?} -> {:?}", seq, new_next);
            
            for n in new_next.into_iter() {
                visited.insert(n.clone());
                q.push_back(n);
            }
        }

        println!("visited: {:?}", visited.len());
        visited.into_iter().map(|n| n.as_ref().clone()).collect_vec()
    }
    pub fn get_init_config(&self) -> Vec<PCPConfig> {
        PCPConfig {
            seq: "".to_string(),
            dir: PCPDir::UP,
        }.apply_pcp(self)
    }
}
