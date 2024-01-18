use regex::Regex;


#[derive(Debug)]
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

#[derive(Debug)]
pub struct PCP {
    pub tiles: Vec<Tile>,
}

impl PCP {
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
