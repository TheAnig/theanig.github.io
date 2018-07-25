Gem::Specification.new do |s|
  s.name     = 'julia-jekyll-theme'
  s.version  = '0.2.1'
  s.license  = 'MIT'
  s.summary  = 'A minimalistic jekyll theme'
  s.author   = 'TheAnig'
  s.email    = 'theanig@protonmail.com'
  s.homepage = 'https://github.com/TheAnig/julia'
  s.files    = `git ls-files -z`.split("\x0").grep(%r{^_(sass|includes|layouts)/})
end
